import platform

import matplotlib as mpl
import scipy as sp
from matplotlib.legend_handler import HandlerLine2D

from relnet.agent.mcts.il.imitation_learning import ImitationLearningAgent
from relnet.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent

from relnet.agent.baseline.best_response_agent import PayoffTransferAgent, BestResponseAgent

from relnet.agent.baseline.simulated_annealing import SimulatedAnnealingAgent


from relnet.agent.baseline.baseline_agent import *
from relnet.objective_functions.objective_functions import SocialWelfare, MeanCost, Fairness
from relnet.state.network_generators import *
from relnet.agent.mcts.il.imitation_learning import *

if platform.system() == 'Darwin':
    mpl.use("TkAgg")
import matplotlib.animation
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np

from itertools import product

agent_display_names = {
                      RandomAgent.algorithm_name: "Rand",
                       TargetHubsAgent.algorithm_name: "TH",
                       TargetMinCostAgent.algorithm_name: "TLC",
                       ExhaustiveSearchAgent.algorithm_name: "ES",
                       BestResponseAgent.algorithm_name: "BR",
                       PayoffTransferAgent.algorithm_name: "PT",
                       SimulatedAnnealingAgent.algorithm_name: "SA",
                       MonteCarloTreeSearchAgent.algorithm_name: "UCT",
                       ImitationLearningAgent.algorithm_name: "GIL (ours)"

                       }

objective_function_display_names = {
                                    SocialWelfare.name: "SW",
                                    MeanCost.name: "MC",
                                    Fairness.name: "F"
                                    }

network_generator_display_names = {
                                   GNMNetworkGenerator.name: "Erdős–Rényi",
                                   BANetworkGenerator.name: "Barabási–Albert",
                                   WattsStrogatzNetworkGenerator.name: "Watts-Strogatz",
                                   RegularNetworkGenerator.name: "Regular"

                                   }

setting_display_names = {
    "hc": "HC",
    "ic": "IC"
}

plot_objective_display_names = {"ms_per_move": "Mean milliseconds per move",
                                "cummulative_reward": "Cumulative reward"
                                }

plot_objective_legend_positions = {"ms_per_move": "upper right",
                                   "cummulative_reward": "upper right"
                                   }

use_latex = True
fig_dpi = 200



def animate_episode(state_history, show_animation=True, save_animation=False, animation_filename='animation.htm'):
    fig, ax = plt.subplots(figsize=(8, 8), clear=True)

    def update(i):
        ax.clear()
        state = state_history[i]
        state.display(ax)

    ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(state_history), interval=1000, repeat=True)
    if save_animation:
        # HTML(ani.to_jshtml())
        ani.save(animation_filename, writer='html', fps=1)
    if show_animation:
        plt.show()
    return ani

def set_latex_if_required():
    if use_latex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.unicode'] = True

def plot_eval_histories(results_df,
                      figure_save_path):
    sns.set(font_scale=3.5)
    plt.rcParams["lines.linewidth"] = 3.5
    plt.rc('font', family='serif')
    set_latex_if_required()

    # dims = (16.54, 24.81)
    # dims = (16.54, 16.54)

    settings = results_df["setting"].unique()
    perfs = results_df["objective_function"].unique()
    net_types = results_df["network_generator"].unique()

    problems = list(product(settings, perfs))

    num_problems = len(problems)
    num_net_types = len(net_types)

    dims = (8.26 * num_problems, 18.60)

    fig, axes = plt.subplots(num_net_types, num_problems, sharex='none', sharey='none', figsize=dims)

    for i in range(num_net_types):
        for j in range(num_problems):
            net_type = net_types[i]
            setting, perf = problems[j]

            filtered_data = results_df[(results_df['network_generator'] == net_type) &
                                       (results_df['setting'] == setting) &
                                       (results_df['objective_function'] == perf)]

            filtered_data = filtered_data.rename(columns={"timestep": "batch number",
                                                          "perf": "Cumulative reward"})

            ax = axes[i][j]
            ax = sns.lineplot(data=filtered_data, x="batch number", y="Cumulative reward",
                              ax=ax)

            handles, labels = ax.get_legend_handles_labels()

            if i < 2:
                ax.set_xlabel('')

            if j == 0:
                ax.set_ylabel('$\mathbf{G}^{eval}$ mean reward', size="small")
            else:
                ax.set_ylabel('')

            #ax.legend_.remove()
            #ax.set_xticks(network_sizes)

    pad = 2.5  # in points

    rows = net_types
    cols = problems

    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{setting_display_names[col[0]]}, {objective_function_display_names[col[1]]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(f"{network_generator_display_names[row]}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    fig.tight_layout()
    # fig.tight_layout(rect=[0,0,1,0.90])
    # fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    # plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0


def plot_size_based_results(results_df,
                      figure_save_path,
                      network_sizes):

    sns.set(font_scale=3.5)
    plt.rcParams["lines.linewidth"] = 3.5
    plt.rc('font', family='serif')
    set_latex_if_required()

    #dims = (16.54, 24.81)
    #dims = (16.54, 16.54)

    settings = results_df["setting"].unique()
    perfs = results_df["objective_function"].unique()
    net_types = results_df["network_generator"].unique()

    problems = list(product(settings, perfs))

    num_problems = len(problems)
    num_net_types = len(net_types)

    dims = (8.26 * num_problems, 18.60)

    fig, axes = plt.subplots(num_net_types, num_problems, sharex='none', sharey='none', figsize=dims)


    for i in range(num_net_types):
        for j in range(num_problems):
            net_type = net_types[i]
            setting, perf = problems[j]

            filtered_data = results_df[(results_df['network_generator'] == net_type) &
                                       (results_df['setting'] == setting) &
                                       (results_df['objective_function'] == perf) &
                                       ~(results_df['algorithm'] == ExhaustiveSearchAgent.algorithm_name)]

            filtered_data = filtered_data.rename(columns={"network_size": "number of players $n$",
                                                          "cummulative_reward": "Cumulative reward"})

            es_data = results_df[(results_df['network_generator'] == net_type) &
                                       (results_df['setting'] == setting) &
                                       (results_df['objective_function'] == perf) &
                                       (results_df['algorithm'] == ExhaustiveSearchAgent.algorithm_name)]

            es_data = es_data.rename(columns={"network_size": "number of players $n$",
                                                          "cummulative_reward": "Cumulative reward"})
            es_means = es_data.mean(axis=0)

            ax = axes[i][j]
            ax = sns.lineplot(data=filtered_data, x="number of players $n$", y="Cumulative reward", hue="algorithm", ax=ax)

            handles, labels = ax.get_legend_handles_labels()
            es_idx = None
            for k, lab in enumerate(labels):
                if str(lab) == ExhaustiveSearchAgent.algorithm_name:
                    es_idx = k
                    break

            if es_idx is not None:
                es_color = [ax.legend_.get_lines()[es_idx].get_color()]
                ax.scatter([es_means["number of players $n$"]], [es_means['Cumulative reward']], marker="*", c=es_color, s=999.)

            if i < 2:
                ax.set_xlabel('')

            if j == 0:
                ax.set_ylabel('$\mathbf{G}^{test}$ mean reward', size="small")
            else:
                ax.set_ylabel('')

            ax.legend_.remove()
            ax.set_xticks(network_sizes)


    display_labels = [agent_display_names[label] for label in labels[1:]]
    #legend_ax.legend(handles[1:], display_labels, loc='lower right', borderaxespad=0.1, fontsize="x-small", ncol=2)


    fig.legend(handles[1:], display_labels, loc='upper center', borderaxespad=-0.075, fontsize="x-small", ncol=9,
               handler_map={plt.Line2D : HandlerLine2D(update_func=legend_handle_update)})

    pad = 2.5  # in points

    rows = net_types
    cols = problems

    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{setting_display_names[col[0]]}, {objective_function_display_names[col[1]]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(f"{network_generator_display_names[row]}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    fig.tight_layout()
    #fig.tight_layout(rect=[0,0,1,0.90])
    #fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    #plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0


def plot_timings(results_df,
                      figure_save_path,
                      network_sizes):

    sns.set(font_scale=3)
    plt.rcParams["lines.linewidth"] = 5
    plt.rc('font', family='serif')
    set_latex_if_required()

    dims = (8.27, 6.89)
    fig, axes = plt.subplots(1, 1, sharex='all', sharey='none', figsize=dims, squeeze=False)
    ax = axes[0][0]
    ax.set_yscale('log')

    results_df = results_df.rename(columns={"network_size": "number of players $n$",
                                      "episode_duration_ms": "Episode duration (ms)"})

    es_data = results_df[results_df['algorithm'] == ExhaustiveSearchAgent.algorithm_name]
    es_data = es_data.rename(columns={"network_size": "number of players $n$",
                                      "episode_duration_ms": "Episode duration (ms)"})
    es_means = es_data.mean(axis=0)
    print(es_means['Episode duration (ms)'])


    sns.lineplot(data=results_df, x="number of players $n$", y="Episode duration (ms)", hue="algorithm", ax=ax)
    ax.set_xticks(network_sizes)

    handles, labels = ax.get_legend_handles_labels()
    es_idx = None
    for k, lab in enumerate(labels):
        if str(lab) == ExhaustiveSearchAgent.algorithm_name:
            es_idx = k
            break

    if es_idx is not None:
        es_color = [ax.legend_.get_lines()[es_idx].get_color()]
        ax.scatter([es_means["number of players $n$"]], [es_means['Episode duration (ms)']], marker="*", c=es_color, s=333.)

    ax.legend_.remove()

    display_labels = [agent_display_names[label] for label in labels[1:]]
    fig.legend(handles[1:], display_labels, loc='upper center', borderaxespad=-0.1, fontsize="xx-small", ncol=3,
               handler_map={plt.Line2D: HandlerLine2D(update_func=legend_handle_update)})

    # fig.tight_layout()
    fig.tight_layout(rect=[0,0,1,0.92])

    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0


def plot_il_method_data(results_df,
                      figure_save_path):

    sns.set(font_scale=3.5)
    plt.rcParams["lines.linewidth"] = 3.5
    plt.rc('font', family='serif')
    plt.rcParams['hatch.linewidth'] = 0.075


    set_latex_if_required()


    settings = results_df["setting"].unique()
    perfs = results_df["objective_function"].unique()
    net_types = results_df["network_generator"].unique()
    il_procs = results_df["il_procedure"].unique()


    problems = list(product(settings, perfs))

    num_problems = len(problems)
    num_net_types = len(net_types)
    num_sizes = len(results_df["network_size"].unique())

    dims = (8.26 * num_problems, 16.54)
    # dims = (16.54, 24.81)
    # dims = (16.54, 16.54)

    ylimits = {}
    for net_type in net_types:
        proc_raw = results_df[results_df['network_generator'] == net_type]
        proc_data = pd.pivot_table(proc_raw, values='avg_reward',
                       index=['network_generator', 'setting', 'objective_function'],
                       aggfunc=np.mean)['avg_reward']
        min_val = proc_data.min()
        max_val = proc_data.max()

        ylim_inf = min_val - (min_val / 7.5)
        #ylim_sup = max_val + (max_val / 20)

        ylimits[net_type] = ylim_inf
        #ylimits[net_type] = (ylim_inf, ylim_sup)



    fig, axes = plt.subplots(num_net_types, num_problems, sharex='none', sharey='row', figsize=dims)



    for i in range(num_net_types):
        for j in range(num_problems):
            net_type = net_types[i]
            setting, perf = problems[j]

            filtered_data = results_df[(results_df['network_generator'] == net_type) &
                                       (results_df['setting'] == setting) &
                                       (results_df['objective_function'] == perf)]

            filtered_data = filtered_data.rename(columns={"avg_reward": "Cumulative reward",
                                                    "il_procedure": "IL Method",
                                                    "network_size": "number of players $n$"})
            filtered_data.replace(agent_display_names, inplace=True)

            ax = axes[i][j]
            ax = sns.barplot(data=filtered_data, x="number of players $n$", y="Cumulative reward", hue='IL Method', ax=ax)
            ax.set_ylim(ylimits[net_type])

            hatches = itertools.cycle(['//', '+', 'o'])
            for k, bar in enumerate(ax.patches):
                if k % num_sizes == 0:
                    hatch = next(hatches)
                bar.set_hatch(hatch)
                bar.set_edgecolor('k')

            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

            if i < 2:
                ax.set_xlabel('')

            if j == 0:
                ax.set_ylabel('$\mathbf{G}^{eval}$ mean reward', size="small")
            else:
                ax.set_ylabel('')

    fig.legend(handles, labels, loc='upper center', borderaxespad=-0.25, fontsize="small", ncol=4,
               handler_map={plt.Line2D: HandlerLine2D(update_func=legend_handle_update)})


    pad = 2.5  # in points

    rows = net_types
    cols = problems

    for ax, col in zip(axes[0], cols):
        ax.annotate(f"{setting_display_names[col[0]]}, {objective_function_display_names[col[1]]}",
                    xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='medium', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(f"{network_generator_display_names[row]}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    rotation=90,
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='medium', ha='right', va='center')

    fig.tight_layout()
    fig.savefig(figure_save_path, bbox_inches='tight', dpi=fig_dpi)

    plt.show()
    plt.close()
    plt.rcParams["lines.linewidth"] = 1.0

def legend_handle_update(handle, orig):
    handle.update_from(orig)
    handle.set_linewidth(5)


def get_annotation_box_width(fig, annotation):
    extent = annotation.get_window_extent(renderer=fig.canvas.get_renderer())
    x0, x1 = extent.x0, extent.x1
    size_pixels = x1 - x0
    size_inches = size_pixels / fig_dpi
    return size_inches

def ci_aggfunc(series):
    std = np.std(series)
    return 1.96 * std

def compute_ci(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    se = sp.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n-1)
    return h