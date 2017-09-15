from itertools import *
import numpy as np 
import pandas as pd 
import random
import tensorflow as tf
import os
import itertools
from Extract_data_from_database import *

class StateTracker():
	def __init__(self, people_number, question_number):
		self.people_num = people_number
		self.question_num = question_number
		self.database_yes, self.database_no, self.database_unknown, self.database_total = construct_read_database(people_number, question_number)
		self.rows, self.columns = people_number, question_number
	def dialog_initialization(self):
		self.selected_people_no = random.choice(np.arange(self.rows))
		# self.selected_people_info = self.data[self.selected_people_no]
		self.turn = 0
		self.question_list = []
		self.answer_list = []
		self.reward = None
		self.terminal = False
		self.guess_result = 0
		# construct features
		self.asked_question = np.zeros(self.question_num)
		self.prob_of_people = np.ones(self.people_num, dtype = float)
		self.certainty_vector = self.compute_certainty()
		self.entropy_vector = self.compute_entropy()
		self.state_rep = self.state_representation()

		self.state = {'qa_pair': None, 'turn': self.turn, 'reward': None, 'guess': self.guess_result, 'terminal': self.terminal}
	def update(self, agent_action):
		if agent_action == self.question_num:
			# self.guess = True
			self.terminal = True
			guess_people = self.choose_people_statisfied()
			# print("\nselected_number is {}, guessed_number is: {}, candidate people length is: {}\n".format(self.selected_people_no, guess_people, len(satisfied_idx)))
			if guess_people == self.selected_people_no:
				self.reward = 80
				self.guess_result = 1
			else:
				self.reward = -30
				self.guess_result = -1
		else:
			self.turn += 1
			self.reward = -1
			self.question_list.append(agent_action)
			answer = self.answer_question(agent_action)
			self.answer_list.append(answer)

			# Update feature
			self.asked_question[agent_action] = 1
			self.prob_of_people = self.update_probability_of_people(agent_action, answer)
			self.certainty_vector = self.compute_certainty()
			self.entropy_vector = self.compute_entropy()
			self.state_rep = self.state_representation()

			if self.turn > 30:
				# self.guess = True
				self.terminal = True
				guess_people = self.choose_people_statisfied()
				# print("\nselected_number is {}, guessed_number is: {}, candidate people length is: {}\n".format(self.selected_people_no, guess_people, len(satisfied_idx)))
				if guess_people == self.selected_people_no:
					self.reward = 80
					self.guess_result = 1
				else:
					self.reward = -30
					self.guess_result = -1
			else:
				# self.guess = False
				self.terminal = False
		self.state = {'qa_pair': [self.question_list, self.answer_list], 'turn': self.turn, 'reward': self.reward, 'guess': self.guess_result,  'terminal': self.terminal}	
	
	def answer_question(self, agent_action):
		yes_prob = self.database_yes[self.selected_people_no, agent_action]
		no_prob = self.database_no[self.selected_people_no, agent_action]
		unknown_prob = self.database_unknown[self.selected_people_no, agent_action]
		assert((sum([yes_prob, no_prob, unknown_prob]) - 1.0) < 0.0001)
		answer = np.random.choice([1,2,3], p = [yes_prob, no_prob, unknown_prob])
		return answer

	def update_probability_of_people(self, agent_action, answer):
		if answer == 1:
			tmp_prob = (self.database_yes[:, agent_action] + self.database_unknown[:, agent_action] * 0.5)
		elif answer == 2:
			tmp_prob = (self.database_no[:, agent_action] + self.database_unknown[:, agent_action] * 0.5)
		else:
			tmp_prob = self.database_unknown[:, agent_action] * 0.5
		prob_of_people = (self.turn * self.prob_of_people + tmp_prob) / (self.turn + 1)
		return prob_of_people

	def compute_certainty(self):
		certainty_matrix = (1 - self.database_unknown) * self.prob_of_people.reshape(-1,1) 
		certainty_vector = np.average(certainty_matrix, axis = 0)
		return  certainty_vector

	def compute_entropy(self):
		tmp_matrix = ((self.database_yes + 0.5 * self.database_unknown)*10).astype(int)
		tmp_dataframe = pd.DataFrame(tmp_matrix)
		entropy_vector = tmp_dataframe.apply(self.entropy, axis = 0)
		return entropy_vector

	def entropy(self, data_list):
		data_list = data_list.tolist()
		length = len(data_list)
		result = 0
		for i in range(11):
			if data_list.count(i) != 0:
				result -= data_list.count(i)*1.0/length * np.log(data_list.count(i)*1.0/length)
		return result

	def state_representation(self):
		state_rep = np.hstack((self.asked_question, self.prob_of_people, self.certainty_vector, self.entropy_vector, self.turn/20.))
		state_rep = state_rep.reshape(1, -1)
		return state_rep

	def state_for_agent(self):
		return {'terminal': self.terminal, 'representation': self.state_rep}

	def choose_people_statisfied(self):
		# all_satisfied = np.repeat(True, self.rows)
		# if self.turn == 0:
		# 	return list(range(self.people_num)), random.choice(np.arange(self.data.shape[0]))
		# else:
		# 	for question_no, answer in zip(self.state['qa_pair'][0], self.state['qa_pair'][1]):
		# 		right_list = self.data[:, question_no] == answer
		# 		all_satisfied  = [i and j for i, j in zip(all_satisfied, right_list)]
		# 	satisfied_idx = [idx for idx, bool_value in enumerate(all_satisfied) if bool_value]
		# 	return satisfied_idx, random.choice(satisfied_idx)
		return np.argmax(self.prob_of_people)
	# def state_for_agent(self):
	# 	return {'terminal': self.terminal, 'representation': self.state_represent_comp_matrix(self.state)}

	# def state_represent_comp_matrix(self, state):
	# 	# construct features for completion dataset
	# 	# compute how many people satis=fied all the answer
	# 	all_satisfied = np.repeat(True, self.rows)
	# 	asked_question = np.zeros(self.columns)
	# 	if state['turn'] == 0:
	# 		all_satisfied_ratio = 1.0
	# 	else:
	# 		for question_no, answer in zip(state['qa_pair'][0], state['qa_pair'][1]):
	# 			if answer == 1:
	# 				self.database_yes

	# 			right_list = self.data[:, question_no] == answer
	# 			all_satisfied  = [i and j for i, j in zip(all_satisfied, right_list)]
	# 			asked_question[question_no] = 1
	# 	state_rep = np.hstack((asked_question, all_satisfied, state['turn']/20.))
	# 	# state_rep = np.hstack((asked_question, state['turn']/20.))
	# 	state_rep = state_rep.reshape(1, -1)
	# 	return state_rep

class Agent():
	def __init__(self, people_num, question_num, scope = 'estimator'):
		self.question_num = question_num
		self.people_num = people_num
		self.actions_num = self.question_num + 1
		self.scope = scope
		with tf.variable_scope(scope):
			self.build_model()

	def build_model(self):
		self.x = tf.placeholder(shape = [None, self.actions_num + self.people_num + self.question_num + self.question_num], dtype = tf.float32)
		# The TD target value
		self.y = tf.placeholder(shape = [None], dtype = tf.float32)
		# Integer id of which actions was selected
		self.actions = tf.placeholder(shape = [None], dtype = tf.int32)

		batch_size = tf.shape(self.x)[0]
		hidden = tf.contrib.layers.fully_connected(self.x, 128)
		# predict the state-action values
		self.predictions = tf.contrib.layers.fully_connected(hidden, self.actions_num)
		# Get the predictions for the chosen actions only
		gather_indices = tf.range(batch_size)*tf.shape(self.predictions)[1] + self.actions
		self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
		# Calculate the loss
		self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.action_predictions))
		# Optimizer Parameters from original paper
		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
	def predict_q_values(self, sess, state):
		"""
		Predicts state-action values.

		Returns:
		  Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
		  action values.
		"""
		return sess.run(self.predictions, { self.x: state })
	def state_to_action(self, sess, state, epsilon):
		q_values = self.predict_q_values(sess, state)
		actions_prob = np.ones(self.actions_num, dtype = float) * epsilon/self.actions_num
		best_action = np.argmax(q_values)
		actions_prob[best_action] += (1.0 - epsilon)
		action_id = np.random.choice(np.arange(len(actions_prob)), p=actions_prob)
		return action_id
	def update(self, sess, s, a, y):
		"""
		Updates the estimator towards the given targets.

		Args:
		  sess: Tensorflow session object
		  s: State input of shape [batch_size, 4, 160, 160, 3]
		  a: Chosen actions of shape [batch_size]
		  y: Targets of shape [batch_size]

		Returns:
		  The calculated loss on the batch.
		"""
		feed_dict = {self.x: s, self.y: y, self.actions: a}
		global_step, _, loss = sess.run(
			[tf.contrib.framework.get_global_step(), self.train_op, self.loss],
			feed_dict)
		return loss

class ModelParameterCopier():
	"""
	Copy model parameters of one estimator to another
	"""
	def __init__(self, estimator1, estimator2):
		"""
		Defines copy-work operation graph
		"""
		e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
		e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
		e1_params = sorted(e1_params, key=lambda v: v.name)
		e2_params = sorted(e2_params, key=lambda v: v.name)

		self.update_ops = []
		for e1_v, e2_v in zip(e1_params, e2_params):
			op = e2_v.assign(e1_v)
			self.update_ops.append(op)

	def make(self, sess):
		sess.run(self.update_ops)
