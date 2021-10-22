import datetime
import math
import traceback
from copy import copy, deepcopy
from pathlib import Path

import torch
import numpy as np
from relnet.environment.graph_mis_env import GraphMISEnv

from relnet.agent.base_agent import Agent
from relnet.agent.baseline.baseline_agent import RandomAgent
from relnet.evaluation.file_paths import FilePaths
from relnet.utils.config_utils import get_device_placement

class PyTorchAgent(Agent):
    DEFAULT_BATCH_SIZE = 50
    NUM_BASELINE_OBJ_SAMPLES = 100

    def __init__(self, environment):
        super().__init__(environment)

        self.enable_assertions = True
        self.hist_out = None

        self.setup_step_metrics()

    def setup_step_metrics(self):
        self.validation_change_threshold = 0.
        self.best_validation_changed_steps = {}
        self.best_validation_losses = {}
        self.step = 0

    def setup_graphs(self, train_g_list, validation_g_list):
        self.train_g_list = train_g_list
        self.validation_g_list = validation_g_list

        self.train_initial_obj_values = self.get_baseline_obj_values(self.train_g_list, use_zeros=True)
        self.validation_initial_obj_values = self.get_baseline_obj_values(self.validation_g_list, use_zeros=True)

    def get_baseline_obj_values(self, g_list, use_zeros=False):
        all_baseline_vals = np.zeros((self.NUM_BASELINE_OBJ_SAMPLES, len(g_list)), dtype=np.float64)
        if use_zeros:
            return np.mean(all_baseline_vals, axis=0)
        for i in range(self.NUM_BASELINE_OBJ_SAMPLES):
            # print(f"doing baseline estimation number {i + 1}/{self.NUM_BASELINE_OBJ_SAMPLES}.")

            g_list_cp = [deepcopy(g) for g in g_list]
            env_ref = GraphMISEnv.from_env_instance(self.environment)
            randy = RandomAgent(env_ref)

            opts_copy = deepcopy(self.options)
            opts_copy['random_seed'] = opts_copy['random_seed'] * (i+1)

            randy.setup(opts_copy, {})

            env_ref.setup(g_list_cp, np.zeros(len(g_list_cp)), training=False)
            randy.post_env_setup()

            t = 0
            while not env_ref.is_terminal():
                # print(f"making actions at time {t}.")
                list_at = randy.make_actions(t)
                # print(f"at step {t} agent picked actions {list_at}")
                env_ref.step(list_at)
                t += 1

            final_obj_values = env_ref.get_final_values()
            all_baseline_vals[i, :] = final_obj_values

        baseline_vals = np.mean(all_baseline_vals, axis=0)
        return baseline_vals


    def setup_sample_idxes(self, dataset_size):
        self.sample_idxes = list(range(dataset_size))
        self.pos = 0

    def advance_pos_and_sample_indices(self):
        if (self.pos + 1) * self.batch_size > len(self.sample_idxes):
            self.pos = 0
            np.random.shuffle(self.sample_idxes)

        selected_idx = self.sample_idxes[self.pos * self.batch_size: (self.pos + 1) * self.batch_size]
        self.pos += 1
        return selected_idx

    def save_model_checkpoints(self, model_suffix=None):
        model_path = self.get_model_path(model_suffix, init_dir=True)
        torch.save(self.net.state_dict(), model_path)

    def restore_model_from_checkpoint(self, model_suffix=None):
        model_path = self.get_model_path(model_suffix, init_dir=True)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint)

    def get_model_path(self, model_suffix, init_dir=False):
        model_dir = self.checkpoints_path / self.model_identifier_prefix
        if init_dir:
            model_dir.mkdir(parents=True, exist_ok=True)

        if model_suffix is None:
            model_path = model_dir / f"{self.algorithm_name}_agent.model"
        else:
            model_path = model_dir / f"{self.algorithm_name}_agent_{model_suffix}.model"

        return model_path

    def check_validation_loss_if_req(self, step_number, max_steps,
                                     make_action_kwargs=None,
                                     model_tag='default',
                                     save_model_if_better=True,
                                     save_with_tag=False):

        if step_number % self.validation_check_interval == 0 or step_number == max_steps:
            self.check_validation_loss(step_number, max_steps,
                                       make_action_kwargs,
                                       model_tag,
                                       save_model_if_better,
                                       save_with_tag)

    def check_validation_loss(self, step_number, max_steps,
                              make_action_kwargs=None,
                              model_tag='default',
                              save_model_if_better=True,
                              save_with_tag=False):

        if model_tag not in self.best_validation_changed_steps:
            self.best_validation_changed_steps[model_tag] = -1
            self.best_validation_losses[model_tag] = float("inf")

        validation_loss = self.log_validation_loss(step_number, model_tag, make_action_kwargs=make_action_kwargs)
        if self.log_progress: self.logger.info(
            f"{model_tag if model_tag != 'default' else 'model'} validation loss: {validation_loss: .4f} at step "
            f"{step_number}.")

        if (self.best_validation_losses[model_tag] - validation_loss) >= self.validation_change_threshold:
            if self.log_progress: self.logger.info(
                f"rejoice! found a better validation loss for model {model_tag} at step {step_number}.")
            self.best_validation_changed_steps[model_tag] = step_number
            self.best_validation_losses[model_tag] = validation_loss

            if save_model_if_better:
                if self.log_progress: self.logger.info("saving model since validation loss is equal or better.")
                model_suffix = model_tag if save_with_tag else None
                self.save_model_checkpoints(model_suffix=model_suffix)

    def log_validation_loss(self, step, model_tag, make_action_kwargs=None):
        performance = self.eval(self.validation_g_list,
                                self.validation_initial_obj_values,
                                validation=True,
                                make_action_kwargs=make_action_kwargs)

        max_improvement = self.environment.objective_function.upper_limit
        validation_loss = max_improvement - performance

        if self.log_tf_summaries:
            from tensorflow import Summary
            validation_summary = Summary(value=[
                Summary.Value(tag=f"{model_tag}_validation_loss", simple_value=validation_loss)
            ])
            self.file_writer.add_summary(validation_summary, step)
            try:
                self.file_writer.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush TF data.")
                    self.logger.warn(traceback.format_exc())

        if self.hist_out is not None:
            self.hist_out.write('%d,%s,%.6f\n' % (step, model_tag, performance))
            try:
                self.hist_out.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush evaluation history.")
                    self.logger.warn(traceback.format_exc())

        return validation_loss

    def print_model_parameters(self, only_first_layer=True):
        param_list = self.net.parameters()

        for params in param_list:
            print(params.view(-1).data)
            if only_first_layer:
                break

    def check_stopping_condition(self, step_number, max_steps):
        if step_number >= max_steps \
                or (step_number - self.best_validation_changed_step > self.max_validation_consecutive_steps):
            if self.log_progress: self.logger.info(
                "number steps exceeded or validation plateaued for too long, stopping training.")
            if self.log_progress: self.logger.info("restoring best model to use for predictions.")
            self.restore_model_from_checkpoint()
            return True
        return False

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        if 'validation_check_interval' in options:
            self.validation_check_interval = options['validation_check_interval']
        else:
            self.validation_check_interval = 5

        if 'max_validation_consecutive_steps' in options:
            self.max_validation_consecutive_steps = options['max_validation_consecutive_steps']
        else:
            self.max_validation_consecutive_steps = 200000

        if 'pytorch_full_print' in options:
            if options['pytorch_full_print']:
                torch.set_printoptions(profile="full")

        if 'enable_assertions' in options:
            self.enable_assertions = options['enable_assertions']

        if 'model_identifier_prefix' in options:
            self.model_identifier_prefix = options['model_identifier_prefix']
        else:
            self.model_identifier_prefix = FilePaths.DEFAULT_MODEL_PREFIX

        if 'restore_model' in options:
            self.restore_model = options['restore_model']
        else:
            self.restore_model = False

        if 'models_path' in options:
            self.models_path = Path(options['models_path'])
        else:
            self.models_path = Path.cwd() / FilePaths.MODELS_DIR_NAME

        self.checkpoints_path = self.models_path / FilePaths.CHECKPOINTS_DIR_NAME

        if 'log_tf_summaries' in options and options['log_tf_summaries']:
            self.summaries_path = self.models_path / FilePaths.SUMMARIES_DIR_NAME

            from tensorflow import Graph
            from tensorflow.summary import FileWriter
            self.log_tf_summaries = True
            summary_run_dir = self.get_summaries_run_path()
            self.file_writer = FileWriter(summary_run_dir, Graph())
        else:
            self.log_tf_summaries = False

        if 'batch_size' in hyperparams:
            self.batch_size = hyperparams['batch_size']
        else:
            self.batch_size = self.DEFAULT_BATCH_SIZE

    def get_summaries_run_path(self):
        return self.summaries_path / f"{self.model_identifier_prefix}-summaries"

    def setup_histories_file(self):
        self.eval_histories_path = self.models_path / FilePaths.EVAL_HISTORIES_DIR_NAME
        model_history_filename = self.eval_histories_path / FilePaths.construct_history_file_name(
            self.model_identifier_prefix)
        model_history_file = Path(model_history_filename)
        if model_history_file.exists():
            model_history_file.unlink()
        self.hist_out = open(model_history_filename, 'a')

    def finalize(self):
        if self.hist_out is not None and not self.hist_out.closed:
            self.hist_out.close()
        if self.log_tf_summaries:
            self.file_writer.close()
