from collections import deque, namedtuple
import numpy as np 
import os
from dialog_component_real_data import *
import itertools
import matplotlib.pyplot as plt
from IPython import display
import pylab as pl
def simulation_and_training(state_tracker,
							estimator_policy,
							estimator_value,
							num_episodes,
							experiment_dir,
							discount_factor = 0.99):
	Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

	episode_lengths = np.zeros(num_episodes)
	episode_rewards = np.zeros(num_episodes)
	episode_rewards_incre = []
	success_ratio_for_each_100_epoch = []
	episode_guess = np.zeros(num_episodes)
	episode_turn = np.zeros(num_episodes)
	total_t = 0


	for i_episode in range(num_episodes):
		# saver.save(tf.get_default_session(), checkpoint_path)
		state_tracker.dialog_initialization()
		state_for_agent = state_tracker.state_for_agent()
		state_representation = state_for_agent['representation']
		for t in itertools.count():

			# Take action
			# if state_tracker.turn > 40:
			# 	# action_probs = estimator_policy.predict(state_representation)
			# 	action_id = state_tracker.question_num
			# else:
			action_probs = estimator_policy.predict(state_representation)
			action_id = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			
			# update environment
			state_tracker.update(action_id)
			reward = state_tracker.state['reward']
			done = state_tracker.state['terminal']

			next_state_for_agent = state_tracker.state_for_agent()
			next_state_representation = next_state_for_agent['representation']

			# Update statistics
			episode_rewards[i_episode] += reward
			episode_lengths[i_episode] = t

			# TD Update
			value_next = estimator_value.predict(next_state_representation)
			td_target = reward + discount_factor * value_next
			td_error = td_target - estimator_value.predict(state_representation)

			# Update the value estimator
			estimator_value.update(state_representation, td_target)

			# Update the policy estimator
			# using the td error as our advantage estimate
			estimator_policy.update(state_representation, td_error, action_id)


			if done:
				episode_guess[i_episode] = state_tracker.state['guess']
				episode_turn[i_episode] = state_tracker.state['turn']
				if i_episode % 10 == 0:
					print("Step {} ({}) @ Episode {}/{}, result: {}".format(episode_turn[i_episode], total_t, i_episode + 1, num_episodes, episode_guess[i_episode]))

				if i_episode % 100 == 0 and i_episode > 100:
					print(td_target)
					sucess = sum([result == 1 for result in episode_guess[i_episode-100:i_episode]])
					success_ratio_for_each_100_epoch.append(sucess*0.01)
					pl.figure()
					plt.plot(success_ratio_for_each_100_epoch)
					display.display(pl.gcf())
				break
			state_representation = next_state_representation
			total_t += 1

	return episode_lengths, episode_rewards, episode_turn, episode_guess
