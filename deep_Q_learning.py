import tensorflow as tf 
from collections import deque, namedtuple
import numpy as np 
import os
from dialog_system_3 import *
import itertools
def deep_q_learning(sess,
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

	estimator_copy = ModelParameterCopier(q_estimator, target_estimator)

	# stats = plotting.EpisodeStats(
	# 	episode_lengths=np.zeros(num_episodes),
	# 	episode_rewards=np.zeros(num_episodes))
	episode_lengths=np.zeros(num_episodes)
	episode_rewards=np.zeros(num_episodes)

	# For 'system/' summaries, usefull to check if currrent process looks healthy
	# current_process = psutil.Process()
	
	# Create directories for checkpoints and summaries
	checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
	checkpoint_path = os.path.join(checkpoint_dir, "model")
	monitor_path = os.path.join(experiment_dir, "monitor")

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(monitor_path):
		os.makedirs(monitor_path)

	saver = tf.train.Saver()
	# Load a previous checkpoint if we find one
	# last_checkpoint = tf.train.last_checkpoint(checkpoint_dir)
	# if last_checkpoint:
	# 	print("Loading model checkpoint {} ...\n".format(last_checkpoint))
	# 	saver.restore(sess, last_checkpoint)

		# Get the current time step
	total_t = sess.run(tf.contrib.framework.get_global_step())

	# The epsilon decay schedule
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	state_tracker.initialization()
	state_agent = state_tracker.state_for_agent()
	# print(state_agent)

	for i in range(replay_memory_init_size):
		if state_agent['guess'] == False:
			representation = state_agent['representation']
			action_id = q_estimator.state_to_action1(sess, representation, epsilons[min(total_t, epsilon_decay_steps-1)])
			# print('Turn {} continue ask question: question_id is {}'.format(state_tracker.state['turn'], action_id))
			state_tracker.update1(action_id)
		else:
			representation = state_agent['representation']
			action_id = q_estimator.state_to_action2(sess, representation, epsilons[min(total_t, epsilon_decay_steps-1)])
			# print('guess people: {}'.format(action_id))
			state_tracker.update2(action_id)
		next_state_agent = state_tracker.state_for_agent()
		reward = state_tracker.state['reward']
		# print('reward is {}'.format(reward))
		done = state_tracker.state['terminal']
		# print('Done: {}'.format(done))
		replay_memory.append(Transition(state_agent, action_id, reward, next_state_agent, done))
		if state_tracker.state['terminal']:
			# print('\nRESTART')
			state_tracker.initialization()
			state_agent = state_tracker.state_for_agent()
		else:
			state_agent = next_state_agent

	for i_episode in range(num_episodes):
		saver.save(tf.get_default_session(), checkpoint_path)
		state_tracker.initialization()
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
			if state_agent['guess'] == False:
				representation = state_agent['representation']
				action_id = q_estimator.state_to_action1(sess, representation, epsilon)
				# print('Turn {} continue ask question: question_id is {}'.format(state_tracker.state['turn'], action_id))
				state_tracker.update1(action_id)
			else:
				representation = state_agent['representation']
				action_id = q_estimator.state_to_action2(sess, representation, epsilon)
				# print('guess people: {}'.format(action_id))
				state_tracker.update2(action_id)
			next_state_agent = state_tracker.state_for_agent()
			reward = state_tracker.state['reward']
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
			
			batch_guess_bool = [state_agent['guess'] for state_agent in states_batch] 
			
			# print(guess_done_batch)
			# print(ask_action_batch.shape)
			# print(np.asarray(ask_states_batch).shape)
			
			if sum(batch_guess_bool) != batch_size:
				ask_states_batch = np.asarray([state_agent['representation'] for i, state_agent in enumerate(states_batch) if not batch_guess_bool[i]]).reshape((batch_size - sum(batch_guess_bool), -1))
				ask_action_batch = np.asarray([action for i, action in enumerate(action_batch) if not batch_guess_bool[i]])
				ask_reward_batch = np.asarray([reward for i, reward in enumerate(reward_batch) if not batch_guess_bool[i]])
				ask_next_states_batch = np.asarray([next_state_agent['representation'] for i, next_state_agent in enumerate(next_states_batch) if not batch_guess_bool[i]]).reshape((batch_size - sum(batch_guess_bool), -1))
				ask_done_batch = np.asarray([done for i, done in enumerate(done_batch) if not batch_guess_bool[i]])

				ask_q_values_next = target_estimator.predict_q_values1(sess, ask_next_states_batch)
				ask_targets_batch = ask_reward_batch + np.invert(ask_done_batch).astype(np.float32) * discount_factor * np.amax(ask_q_values_next, axis=1)
				# Perform gradient descent update
				loss1 = q_estimator.update1(sess, ask_states_batch, ask_action_batch, ask_targets_batch)
				loss += loss1

			if sum(batch_guess_bool) != 0:
				guess_states_batch = np.asarray([state_agent['representation'] for i, state_agent in enumerate(states_batch) if batch_guess_bool[i]]).reshape((sum(batch_guess_bool), -1))
				guess_action_batch = np.asarray([action for i, action in enumerate(action_batch) if batch_guess_bool[i]])
				guess_reward_batch = np.asarray([reward for i, reward in enumerate(reward_batch) if batch_guess_bool[i]])
				guess_next_states_batch = np.asarray([next_state_agent['representation'] for i, next_state_agent in enumerate(next_states_batch) if batch_guess_bool[i]]).reshape((sum(batch_guess_bool), -1))
				guess_done_batch = np.asarray([done for i, done in enumerate(done_batch) if batch_guess_bool[i]])

				guess_q_values_next = target_estimator.predict_q_values2(sess, guess_next_states_batch)
				guess_targets_batch = guess_reward_batch + np.invert(guess_done_batch).astype(np.float32) * discount_factor * np.amax(guess_q_values_next, axis=1)
				# Perform gradient descent update
				loss2 = q_estimator.update2(sess, guess_states_batch, guess_action_batch, guess_targets_batch)
				loss += loss2

			if done:
				break
			state_agent = next_state_agent
			total_t += 1
			# Add summaries to tensorboard
		episode_summary = tf.Summary()
		episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
		episode_summary.value.add(simple_value=episode_rewards[i_episode], tag="episode/reward")
		episode_summary.value.add(simple_value=episode_lengths[i_episode], tag="episode/length")
		# episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
		# episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memeory_usage_percent")
		q_estimator.summary_writer.add_summary(episode_summary, i_episode)
		q_estimator.summary_writer.flush()

		# yield total_t, plotting.EpisodeStats(
		# 	episode_lengths=episode_lengths[:i_episode+1],
		# 	episode_rewards=episode_rewards[:i_episode+1])
		yield total_t, episode_lengths[:i_episode+1], episode_rewards[:i_episode+1]

	return episode_lengths, episode_rewards