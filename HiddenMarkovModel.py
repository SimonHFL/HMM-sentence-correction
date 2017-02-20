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

	def get_emission_prob(self, hidden_state, observation):
		distance = levenshtein.distance(hidden_state, observation)
		return poisson.poisson_distribution(0.1, distance)

	def backpointer_to_states(self, backpointer, time, idx):
		if time < 0:
			return ""
		else:
			state = self.states[idx]
			idx = int(backpointer[idx, time])
			return  self.backpointer_to_states(backpointer, time-1, idx) + " " + state

	def viterbi(self, observations):

		# create path probability matrix
		path_probs = np.zeros((len(self.states), len(observations)))
		# create backpointer matrix
		backpointer = np.zeros((len(self.states), len(observations)))

		# initialization step 
		for state_idx, state in enumerate(self.states):
			transition_prob = self.transition_probs[self.start_state_idx, state_idx]
			emission_prob = self.get_emission_prob(state, observations[0])
			path_probs[state_idx, 0] =  transition_prob * emission_prob
			backpointer[state_idx, 0] = self.start_state_idx

		# recursion step
		for obs_idx, observation in enumerate(observations):

			if obs_idx == 0: # we already delt with the first observation
				continue 

			print("observation idx " + str(obs_idx))
			
			for state_idx, state in enumerate(self.states):

				# print progress
				if state_idx % 1000 == 0:
					print("state % " + str(state_idx/len(self.states)))#, end='\r')

				emission_prob = self.get_emission_prob(state, observation)

				# find the most likely path to this state at this time
				max_path_prob = -1
				max_prev_state_idx = -1
				for prev_state_idx, prev_state in enumerate(self.states):
					transition_prob = self.transition_probs[prev_state_idx, state_idx]	
					if transition_prob == 0:
						continue	
					path_prob = path_probs[prev_state_idx, obs_idx-1] * transition_prob * emission_prob
					if(path_prob > max_path_prob):
						max_path_prob = path_prob
						max_prev_state_idx = prev_state_idx

				path_probs[state_idx, obs_idx] = max_path_prob
				backpointer[state_idx, obs_idx] = max_prev_state_idx
			
		# termination step
		max_final_path_prob = -1
		max_final_state_idx = -1
		for state_idx, state in enumerate(self.states):
			final_path_prob = path_probs[state_idx, len(observations) - 1] * self.transition_probs[state_idx, self.end_state_idx]
			if final_path_prob > max_final_path_prob:
				max_final_path_prob = final_path_prob
				max_final_state_idx = state_idx
		print("end max prob")
		print(max_final_path_prob)
		path_probs[self.end_state_idx, len(observations) - 1] = max_final_path_prob
		backpointer[self.end_state_idx, len(observations) - 1] = max_final_state_idx

		return self.backpointer_to_states(backpointer, len(observations)-1, max_final_state_idx)