import pandas as pd
import numpy as np
import levenshtein
import poisson
import sys

class HiddenMarkovModel(object):

	def __init__(self, states, transition_probs, start_state_idx, end_state_idx):
		self.states = states
		self.transition_probs = transition_probs
		self.start_state_idx = start_state_idx
		self.end_state_idx = end_state_idx

	def get_future_states_and_transition_probs(self, state):
		state_idx = self.states.index(state)
		future_state_idxs = np.where( self.transition_probs[state_idx] != 0)[0]
		return [ (self.states[i], self.transition_probs[state_idx, i]) for i in future_state_idxs ]

	def get_final_transition_prob(self, state):
		return self.transition_probs[self.states.index(state), self.end_state_idx]

	def trellis_to_states(self, trellis, state, time):
		if time == 0:
			return ""
		else: 
			return self.trellis_to_states(trellis, trellis[time][state][1], time-1) + " " + state

	def viterbi(self, observations, max_error_rate):

		trellis = [ { '<s>': (1., None) } ]

		# fill out trellis
		for obs_idx, observation in enumerate(observations):
			
			current_states = {}
			trellis.append(current_states)

			# set the maximum amount of errors that are allowed
			max_errors = int(len(observation) * max_error_rate) + 1  

			for prev_state in trellis[obs_idx].keys():

				prev_path_prob = trellis[obs_idx][prev_state][0]

				for future_state, transition_prob in self.get_future_states_and_transition_probs(prev_state):

					distance = levenshtein.distance(future_state, observation)

					# if the levenshtein distance exceeds the maximum, 
					# we disregard the path
					if distance > max_errors: 
						continue

					emission_prob = poisson.poisson_distribution(0.1, distance)

					path_prob = prev_path_prob * transition_prob * emission_prob

					# only keep path with max probability
					if future_state in current_states:
						if current_states[future_state][0] >= path_prob:
							continue

					current_states[future_state] = (path_prob, prev_state)
				

		# termination step
		max_final_path_prob = -1
		max_final_state = None
		for state in trellis[-1].keys():
			final_path_prob = trellis[-1][state][0] * self.get_final_transition_prob(state)
			if final_path_prob > max_final_path_prob:
				max_final_path_prob = final_path_prob
				max_final_state = state

		return self.trellis_to_states(trellis, max_final_state, len(trellis) - 1)
