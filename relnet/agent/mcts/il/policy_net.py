import sys
sys.path.append('/usr/lib/pytorch_structure2vec/s2v_lib')

from torch import nn as nn
from torch.nn import functional as F

import numpy as np
import torch
import torch.nn as nn
from relnet.utils.config_utils import get_device_placement
from pytorch_util import weights_init
from relnet.state.graph_embedding import EmbedMeanField, EmbedLoopyBP


class BasePolicyNet(nn.Module):
    def __init__(self, hyperparams, num_node_feats, num_edge_feats):
        super().__init__()

        self.hyperparams = hyperparams
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        if hyperparams['embedding_method'] == 'mean_field':
            model = EmbedMeanField
        elif hyperparams['embedding_method'] == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            raise ValueError(f"unknown embedding method {hyperparams['embedding_method']}")

        self.embed_dim = hyperparams['latent_dim']
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        weights_init(self)



        self.s2v = model(latent_dim=self.embed_dim,
                         output_dim=0,
                         num_node_feats=self.num_node_feats,
                         num_edge_feats=self.num_edge_feats,
                         max_lv=hyperparams['max_lv'])

    def get_num_nodes_prefix_sum(self, states):
        n_nodes = 0
        num_nodes_prefix_sum = []
        for graph in states:
            n_nodes += graph.num_nodes
            num_nodes_prefix_sum.append(n_nodes)

        return n_nodes, num_nodes_prefix_sum

    def prepare_node_features(self, states):
        n_nodes, nn_prefix_sum = self.get_num_nodes_prefix_sum(states)
        node_feat = torch.zeros(n_nodes, self.num_node_feats)

        for i in range(0, len(nn_prefix_sum)):
            offset = nn_prefix_sum[i-1] if i > 0 else 0
            num_nodes = nn_prefix_sum[i] - nn_prefix_sum[i - 1] if i > 0 else nn_prefix_sum[0]

            selected_nodes = states[i].selected_nodes
            non_selected_nodes = states[i].all_nodes_set - selected_nodes

            selected_idxes = np.array(list(selected_nodes)) + offset
            non_selected_idxes = np.array(list(non_selected_nodes)) + offset

            node_feat[non_selected_idxes, 0] = 0.0
            node_feat[non_selected_idxes, 1] = 1.0

            node_feat[selected_idxes, 0] = 1.0
            node_feat[selected_idxes, 1] = 0.0

            node_feat[offset: offset+num_nodes, 2] = torch.from_numpy(states[i].effort_costs)

        return node_feat, torch.LongTensor(nn_prefix_sum)


    def prepare_actions(self, actions):
        n_actions = 0
        acts_flattened = []
        num_actions_prefix_sum = []

        for act_list in actions:
            n_actions += len(act_list)
            num_actions_prefix_sum.append(n_actions)
            acts_flattened.extend(act_list)

        acts = torch.LongTensor(acts_flattened)
        num_actions_prefix_sum = torch.LongTensor(num_actions_prefix_sum)
        return acts, num_actions_prefix_sum

    def get_act_arr_indices(self, acts, n_nodes_prefix_sum, n_actions_prefix_sum):
        n_nodes_prefix_sum = n_nodes_prefix_sum.data.cpu().numpy()
        n_actions_prefix_sum = n_actions_prefix_sum.data.cpu().numpy()

        act_arr_indices = []

        for i in range(0, len(n_nodes_prefix_sum)):
            # can be done in a one-liner with .roll(), but constrained to old version of PyTorch...
            if i == 0:
                offset = 0
                acts_arr_offset = 0
                num_actions = n_actions_prefix_sum[0]
            else:
                offset = n_actions_prefix_sum[i - 1]
                acts_arr_offset = n_nodes_prefix_sum[i - 1]
                num_actions = n_actions_prefix_sum[i] - n_actions_prefix_sum[i - 1]

            # can also be achieved with .index_add, but docs mention it might introduce nondeterministic behaviour
            for j in range(num_actions):
                act = acts[j + offset].item()
                act_arr_indices.append(act + acts_arr_offset)

        return act_arr_indices

    def get_embeddings(self, states):
        node_feat, n_nodes_prefix_sum = self.prepare_node_features(states)

        if get_device_placement() == 'GPU':
            node_feat = node_feat.cuda()
            n_nodes_prefix_sum = n_nodes_prefix_sum.cuda()

        embed, graph_embed = self.s2v(states, node_feat, None, pool_global=True)

        # embed = embed.cpu()
        # graph_embed = graph_embed.cpu()
        return embed, graph_embed, n_nodes_prefix_sum

    def get_greedy_actions(self, states, allowed_actions):
        reduced_acts = self.get_topk_acts(states, allowed_actions, [1] * len(allowed_actions))
        return [a[0] for a in reduced_acts]

    def process_states_actions(self, actions, states):
        embed, graph_embed, n_nodes_prefix_sum = self.get_embeddings(states)
        acts, n_actions_prefix_sum = self.prepare_actions(actions)
        if get_device_placement() == 'GPU':
            n_actions_prefix_sum = n_actions_prefix_sum.cuda()
        act_arr_indices = self.get_act_arr_indices(acts, n_nodes_prefix_sum, n_actions_prefix_sum)
        acts_embed = embed[act_arr_indices, :]
        graph_embed = graph_embed
        return acts, acts_embed, graph_embed, n_actions_prefix_sum

    def get_topk_acts(self, states, allowed_actions, top_k):
        acts, log_probs, n_actions_prefix_sum = self.forward(states, allowed_actions)
        reduced_acts = []

        for i in range(len(n_actions_prefix_sum)):
            if i==0:
                start_index = 0
            else:
                start_index = n_actions_prefix_sum[i-1]
            end_index = n_actions_prefix_sum[i]

            max_acts = top_k[i]
            action_log_probs = log_probs[start_index:end_index]

            top_log_probs, top_indices = torch.topk(action_log_probs, max_acts)
            reduced_acts.append(acts[start_index:end_index][top_indices].tolist())

        return reduced_acts


class PolicyNet(BasePolicyNet):
    def __init__(self, hyperparams, num_node_feats, num_edge_feats):
        super().__init__(hyperparams, num_node_feats, num_edge_feats)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        weights_init(self)
        temp = hyperparams['softmax_temp']
        self.softmax_temp = nn.Parameter(torch.tensor(temp), requires_grad=True)

    def forward(self, states, actions):
        acts, acts_embed, graph_embed, n_actions_prefix_sum = self.process_states_actions(actions, states)
        log_probs = torch.zeros(acts_embed.size()[0], dtype=torch.float32)

        for i, g_embed in enumerate(graph_embed):
            if i == 0:
                offset = 0
                num_actions = n_actions_prefix_sum[0]
            else:
                offset = n_actions_prefix_sum[i - 1]
                num_actions = n_actions_prefix_sum[i] - n_actions_prefix_sum[i - 1]

            pn_output = self.linear_out(g_embed)

            embedded_acts = acts_embed[offset:offset + num_actions, :]
            similarities = self.compute_embedding_similarities(embedded_acts, pn_output)
            similarities = similarities / self.softmax_temp

            graph_log_probs = F.log_softmax(similarities, dim=0)
            log_probs[offset:offset+num_actions] = graph_log_probs

        if get_device_placement() == 'GPU':
            log_probs = log_probs.cuda()
        return acts, log_probs, n_actions_prefix_sum

    def compute_embedding_similarities(self, embedded_acts, pn_output):
        if self.hyperparams['distance_metric'] == 'euclidean':
            diffs = embedded_acts - pn_output
            dists = torch.norm(diffs, 2, dim=1)
            similarities = (-dists)

        elif self.hyperparams['distance_metric'] == 'cosine':
            similarities = F.cosine_similarity(embedded_acts, pn_output, dim=-1)
        else:
            raise ValueError(f"unknown distance metric {self.hyperparams['distance_metric']}")

        return similarities