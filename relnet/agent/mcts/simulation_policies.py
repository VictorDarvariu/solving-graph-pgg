from copy import copy
from random import Random

import numpy as np


class SimulationPolicy(object):
    def __init__(self, random_state, **kwargs):
        self.local_random = Random()
        self.local_random.setstate(random_state)

    def get_random_state(self):
        return self.local_random.getstate()

class RandomSimulationPolicy(SimulationPolicy):
    def __init__(self, random_state, **kwargs):
        super().__init__(random_state)

    def choose_action(self, state, possible_actions):
        available_acts = tuple(possible_actions)
        chosen_action = self.local_random.choice(available_acts)
        return chosen_action

