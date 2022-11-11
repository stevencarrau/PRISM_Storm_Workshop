from __future__ import division
import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import stormpy.examples
import stormpy.examples.files
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import optimize
from beta_bernoulli import BetaBernoulli
import model_loader
import utils



class UAVModel:
    """docstring for UAVModel

    :param str prism_model: path to the prism model
    :param float epsilon: cumulative density function cut-off
    """
    def __init__(self, prism_model, epsilon):
        self.path = prism_model
        self.parameters, self.prism_model,self.properties = model_loader.load_model(prism_model)
        self.instantiator = stormpy.pars.PMdpInstantiator(self.prism_model)
        self.epsilon = epsilon
        self.beta_gentle = BetaBernoulli(2, 5)
        self.beta_aggressive = BetaBernoulli(2, 5)
        self.bounds_gentle = [0, 1]
        self.bounds_aggressive = [0, 1]
        self.expected_values = []
        self.decisions = []

    def compute(self, s_in):
        point = {}
        for p in self.parameters:
            point[p] = s_in
        rational_parameter_assignments = dict([[x, stormpy.RationalRF(val)] for x, val in point.items()])
        instantiated_model = self.instantiator.instantiate(rational_parameter_assignments)
        result = stormpy.model_checking(instantiated_model,
                                        self.properties[0],
                                        extract_scheduler=True)
        output_value = result.at(0)
        return output_value, result.scheduler, instantiated_model

    def update_model(self,new_values):
        parameters, model, properties = model_loader.reload_model(self.path,new_values[0],
                                                                  new_values[1],
                                                                  new_values[2],
                                                                  new_values[3])
        self.parameters = parameters
        self.prism_model = model
        self.properties = properties
        self.instantiator = stormpy.pars.PMdpInstantiator(self.prism_model)

    def update_bounds(self, beta_distr, both=True):
        """
        :param BetaBernoulli beta_distr
        :param bool both: if False only the upper bound is updated (VaR).
        """
        value_at_risk = beta_distr.value_at_risk(self.epsilon)
        if both:
            left_bound = beta_distr.ppf(1-self.epsilon)
            return [left_bound, value_at_risk]
        else:
            return [0, value_at_risk]

    def update_betas(self, maneuver, prev_tup, new_tup):
        """
        if state 'damage' does not change -> failure
        if state 'damage' changes -> success
        """
        if maneuver == 'aggressive':
            if new_tup[2] == prev_tup[2]:
                self.beta_aggressive.update(successes=0, failures=1)
            else:
                self.beta_aggressive.update(successes=1, failures=0)
            if new_tup[3] == prev_tup[3]:
                self.beta_aggressive.update(successes=0, failures=1)
            else:
                self.beta_aggressive.update(successes=1, failures=0)

        if maneuver == 'gentle':
            if new_tup[2] == prev_tup[2]:
                self.beta_gentle.update(successes=0, failures=1)
            else:
                self.beta_gentle.update(successes=1, failures=0)
            if new_tup[3] == prev_tup[3]:
                self.beta_gentle.update(successes=0, failures=1)
            else:
                self.beta_gentle.update(successes=1, failures=0)

    def run(self, known_p):
        """
        :param list known_p: 2D list with known state transitition probability
            for gentle and aggressive, respectively.
        """
        accumulated_reward = 0
        finish_flag = False
        act_path = []
        expected_rewards_worst = []
        expected_rewards_best = []
        storm_state = 0
        prev_state = self.prism_model.state_valuations.get_json(storm_state)
        prev_tup = [int(prev_state['dx']),
                    int(prev_state['dy']),
                    int(prev_state['damage']),
                    int(prev_state['damage2'])]

        while not finish_flag:
            self.bounds_gentle = self.update_bounds(self.beta_gentle, both=False)
            self.bounds_aggressive = self.update_bounds(self.beta_aggressive)
            print('mode = {:.3}'.format(self.beta_aggressive.mode))
            print('up bound = {:.3}'.format(self.bounds_aggressive[1]))

            # Worst-case policy (just right bound)
            value, policy, out_model = self.compute(self.bounds_aggressive[1])
            # Best-case policy (just left bound)
            value_best = self.compute(self.bounds_aggressive[0])[0]

            # Sample for simulator
            value_true, policy_true, out_model = self.compute(known_p[1])
            # Two steps because action type, then gridworld motion
            reward, storm_state, act_list, finish_flag = model_loader.simulate(out_model, policy, 2)

            act_path += act_list
            expected_rewards_worst += [value+accumulated_reward]
            expected_rewards_best += [value_best+accumulated_reward]
            new_states = self.prism_model.state_valuations.get_json(storm_state)
            new_tup = [int(new_states['dx']),
                       int(new_states['dy']),
                       int(new_states['damage']),
                       int(new_states['damage2'])]

            accumulated_reward += reward
            self.update_betas(maneuver=act_list[0],
                              prev_tup=prev_tup,
                              new_tup=new_tup)
            self.update_model(new_tup)
            prev_tup = new_tup

        print(f"Cost: {accumulated_reward}")
        print(f"Final damage: {new_tup[2]}, {new_tup[3]}")
        print(f"Path: {act_path}")
        print(f"Exepected Reward Worst: {expected_rewards_worst}")
        print(f"Exepected Reward Best: {expected_rewards_best}")



uavmodel = UAVModel(prism_model='aircraft.prism',
                    epsilon=0.999)
uavmodel.run(known_p=[0.02, 0.1])
