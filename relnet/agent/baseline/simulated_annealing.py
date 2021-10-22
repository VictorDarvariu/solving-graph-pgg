import numpy as np

from relnet.agent.baseline.heuristic_agent import HeuristicAgent


class SimulatedAnnealingAgent(HeuristicAgent):
    algorithm_name = 'simulated_annealing'
    is_deterministic = False

    def run_heuristic_strategy(self, original_g, i):
        '''
        rough outline as follows:
        - generate a random MIS to start with from graph
        - in a loop until difference is smaller than eps:
            - run improvement procedure
        - get all nodes playing 1 and store them as the actions
        - output action at time [t]
        '''
        g = original_g.copy()
        self.find_random_mis(g)

        old_m = float("+inf")
        t = 1
        steps_without_improvement = 0

        while True:
            current_g = g.copy()
            self.run_improvement_procedure(current_g)

            new_m = current_g.get_number_effort_nodes()

            if new_m < old_m:
                old_m = new_m
                g = current_g
                steps_without_improvement = 0

            else:
                diff = new_m - old_m
                if diff == 0:
                    accepted = False
                else:
                    accept_prob = t ** ((-self.eps) * float(diff))
                    p = [accept_prob, 1-accept_prob]
                    accepted = np.random.choice([True, False], p=p)
                if accepted:
                    old_m = new_m
                    g = current_g
                    steps_without_improvement = 0

                steps_without_improvement += 1

            if steps_without_improvement >= self.c_threshold:
                break

            if t > self.m_threshold:
                break

            t += 1

        end_mis = g.node_labels[g.effort_levels == 1]
        # assert self.is_mis(current_g, current_g.effort_levels)
        return list(end_mis), g

    def run_improvement_procedure(self, current_g):
        '''
        - pick a node playing 0 to flip to 1
        - in a loop until all nodes satisfied:
            - get all nodes that are unsatisfied
                - for best-shot PGG: playing 1 when one of their neighbours already is, or playing 0 and
                none of the neighbours are playing 1
            - randomly pick one unsatisfied node and play best-response strategy: reverse the action
        '''
        no_effort_nodes = current_g.node_labels[(current_g.effort_levels == 0)]
        original_flip = self.local_random.choice(no_effort_nodes)
        current_g.effort_levels[original_flip] = 1

        twoh_neighbours = self.get_twoh_neighbours(current_g, original_flip)
        unsatisfied_nodes, efforts  = self.find_unsatisfied_nodes(current_g, twoh_neighbours)

        while True:
            if len(unsatisfied_nodes) > 0:
                random_idx = self.local_random.randint(0, len(unsatisfied_nodes) - 1)
                node_to_flip = unsatisfied_nodes[random_idx]
                flip_act = int(not efforts[random_idx])
                current_g.effort_levels[node_to_flip] = flip_act
                unsatisfied_nodes, efforts = self.find_unsatisfied_nodes(current_g, twoh_neighbours)
            else:
                #assert self.is_mis(current_g, current_g.effort_levels)
                break

    def get_twoh_neighbours(self, g, node_to_flip):
        twoh_neighbours = set()
        for oneh in g.neighbors[node_to_flip]:
            twoh_neighbours.add(oneh)
            for twoh in g.neighbors[oneh]:
                twoh_neighbours.add(twoh)
        twoh_neighbours.remove(node_to_flip)
        return twoh_neighbours

    def find_random_mis(self, g):
        while len(g.nodes_not_covered) > 0:
            random_node = self.local_random.choice(tuple(g.nodes_not_covered))
            g.select_node(random_node)

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

        if len(hyperparams) == 0:
            self.hyperparams = self.get_default_hyperparameters()

        if 'eps' in self.hyperparams:
            self.eps = self.hyperparams['eps']

        if 'c_threshold' in self.hyperparams:
            self.c_threshold = self.hyperparams['c_threshold']

        if 'm_threshold' in self.hyperparams:
            self.m_threshold = self.hyperparams['m_threshold']

    @classmethod
    def get_default_hyperparameters(cls):
        return {'eps': 10,
                'c_threshold': 10**4,
                'm_threshold': 10**7}