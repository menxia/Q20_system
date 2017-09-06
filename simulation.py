import tensorflow as tf 
from collections import deque, namedtuple
import numpy as np 
import os
from dialog_component_easy_version import *
import itertools
def simulation_and_training(sess,
					state_tracker,
					target_estimator,
					q_estimator,
					num_episodes,
					experiment_dir,
					replay_memory_size = 50000,
					replay_memory_init_size = 500,
					update_target_estimator_every = 1000,
					discount_factor = 0.99,
					epsilon_start = 1.0,
					epsilon_end = 0.1,
					epsilon_decay_steps = 50000,
					batch_size = 64):
	Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
	replay_memory = []

	episode_lengths=np.zeros(num_episodes)
	episode_rewards=np.zeros(num_episodes)

	estimator_copy = ModelParameterCopier(q_estimator, target_estimator)

	# saver = tf.train.Saver()

	# Get the current time step
	total_t = sess.run(tf.contrib.framework.get_global_step())

	# The epsilon decay schedule
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	state_tracker.dialog_initialization()
	state_agent = state_tracker.state_for_agent()
	# print(state_agent)

	for i in range(replay_memory_init_size):
		representation = state_agent['representation']
		action_id = q_estimator.state_to_action(sess, representation, epsilons[min(total_t, epsilon_decay_steps-1)])
		# print('Turn {} continue ask question: question_id is {}'.format(state_tracker.state['turn'], action_id))
		state_tracker.update(action_id)

		next_state_agent = state_tracker.state_for_agent()
		reward = state_tracker.state['reward']
		# print('reward is {}'.format(reward))
		done = state_tracker.state['terminal']
		# print('Done: {}'.format(done))
		replay_memory.append(Transition(state_agent, action_id, reward, next_state_agent, done))
		if state_tracker.state['terminal']:
			# print('\nRESTART')
			state_tracker.dialog_initialization()
			state_agent = state_tracker.state_for_agent()
		else:
			state_agent = next_state_agent

	for i_episode in range(num_episodes):
		# saver.save(tf.get_default_session(), checkpoint_path)
		state_tracker.dialog_initialization()
		state_agent = state_tracker.state_for_agent()
		loss = 0
		for t in itertools.count():
			# Epsilon for this time step
			epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
			# Maybe update the target estimator
			if total_t % update_target_estimator_every == 0:
				print("Copied model parameters to target network.")
				estimator_copy.make(sess)
			# Print out which step we're on, useful for debugging.
			print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(t, total_t, i_episode + 1, num_episodes, loss), end="")			
			representation = state_agent['representation']
			action_id = q_estimator.state_to_action(sess, representation, epsilons[min(total_t, epsilon_decay_steps-1)])
			# print('Turn {} continue ask question: question_id is {}'.format(state_tracker.state['turn'], action_id))
			state_tracker.update(action_id)

			next_state_agent = state_tracker.state_for_agent()
			reward = state_tracker.state['reward']
			# print('reward is {}'.format(reward))
			done = state_tracker.state['terminal']

			if len(replay_memory) == replay_memory_size:
				replay_memory.pop(0)
			replay_memory.append(Transition(state_agent, action_id, reward, next_state_agent, done))

			# Update statistics
			episode_rewards[i_episode] += reward
			episode_lengths[i_episode] = t

			# sample a minibatch from the replay memory
			samples = random.sample(replay_memory, batch_size)
			states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
			states_batch = np.asarray([state_agent['representation'] for state_agent in states_batch]).reshape(batch_size, -1)
			next_states_batch = np.asarray([next_state_agent['representation'] for next_state_agent in next_states_batch]).reshape(batch_size, -1)
			q_values_next = target_estimator.predict_q_values(sess, next_states_batch)
			targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)
			# Perform gradient descent update
			loss1 = q_estimator.update(sess, states_batch, action_batch, targets_batch)
			loss += loss1

			if done:
				break
			state_agent = next_state_agent
			total_t += 1

		yield total_t, episode_lengths[:i_episode+1], episode_rewards[:i_episode+1]

	return episode_lengths, episode_rewards