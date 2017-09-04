from itertools import *
import numpy as np 
import pandas as pd 
import random
import tensorflow as tf
import os
import itertools

def construct_database(question_num, people_num, sparsity):
	data = list(product([1, -1], repeat = question_num))
	data = random.sample(data, people_num)
	data = np.asarray(data)
	data = np.hstack((data, data[:,:5]))
	# maskted_data = np.ma.array(data, mask = np.random.uniform(size = data.shape) < sparsity, fill_value = 0)
	# data_masked = np.ma.getdata(maskted_data)
	data_masked = data.copy()
	data_masked[np.random.uniform(size = data.shape)<sparsity] = 0
	return data_masked, data

class StateTracker():
	def __init__(self, question_num, people_num, sparsity):
		self.question_num = question_num
		self.people_num = people_num
		self.data_masked, self.data = construct_database(question_num, people_num, sparsity)
		self.data_masked = self.data
		self.rows, self.columns = self.data.shape
	def initialization(self):
		self.people_no = random.choice(np.arange(self.data.shape[0]))
		self.people_info = self.data[self.people_no]
		self.turn = 0
		self.question_list = []
		self.answer_list = []
		self.reward = None
		self.terminal = False
		self.guess = False
		self.state = {'qa_pair': None, 'turn': self.turn, 'reward': None, 'guess': self.guess,'terminal': self.terminal}
	def update1(self, agent_action):
		if agent_action == 25:
			self.guess = True
			self.reward = 0
		else:
			self.turn += 1
			self.reward = 0
			self.question_list.append(agent_action)
			self.answer_list.append(self.people_info[agent_action])
			if self.turn > 20:
				self.guess = True
				self.terminal = False
			else:
				self.guess = False
				self.termial = False
		self.state = {'qa_pair': [self.question_list, self.answer_list], 'turn': self.turn, 'reward': self.reward, 'guess': self.guess,  'terminal': self.terminal}	
	def update2(self, agent_action):
		if agent_action != self.people_no:
			self.reward = -10
			self.terminal = True
		else:
			self.reward = 40
			self.terminal = True
		# update sparsity database
		# if self.state['qa_pair'] != []:
		# 	for question_no, answer in zip(self.state['qa_pair'][0], self.state['qa_pair'][1]):
		# 		self.data_masked[self.people_no, question_no] = answer
		self.state = {'qa_pair': [self.question_list, self.answer_list], 'turn': self.turn, 'reward': self.reward, 'guess': self.guess,  'terminal': self.terminal}	


	def state_for_agent(self):
		return {'guess': self.state['guess'], 'representation': self.state_represent(self.state)}

	def state_represent(self, state):
		# state is dictionary {question_no: answer}
		# compute the sparsity of columns
		columns_sparsity = np.zeros((1, self.columns))
		for i in range(self.columns):
			columns_sparsity[0,i] = sum(self.data_masked[:,i] == 0)/float(self.rows)
		# for each slot, compute how many people statisfy the answer, including the unknow cases 
		# compute how many people satisfied all the answer
		candidate_data_for_each_slot = np.zeros((1, self.columns))
		all_satisfied = np.repeat(True, self.rows)
		if state['turn'] == 0:
			all_satisfied_ratio = 1.0
		else:
			for question_no, answer in zip(state['qa_pair'][0], state['qa_pair'][1]):
				unknow_list = self.data_masked[:, question_no] == 0
				right_list = self.data_masked[:, question_no] == answer
				summ_list = [i or j for i, j in zip(unknow_list, right_list)]
				# print(sum(summ_list))
				candidate_data_for_each_slot[0, question_no] = sum(right_list)/float(self.rows)
				all_satisfied  = [i and j for i, j in zip(all_satisfied, summ_list)]
			all_satisfied_ratio = sum(all_satisfied)/self.rows
		# represent the whole the state
		# print(columns_sparsity)
		# print(candidate_data_for_each_slot)
		# print(all_satisfied_ratio)
		# print( state['turn'])
		state_rep = np.hstack((columns_sparsity, candidate_data_for_each_slot, [all_satisfied], [[all_satisfied_ratio, state['turn']/20., state['guess']]]))
		return state_rep 



class Agent():
	def __init__(self, scope = 'estimator', summaries_dir=None):
		self.scope = scope
		self.question_num = 25
		self.people_num = 100
		self.actions_num1 = self.question_num + 1
		self.actions_num2 = self.people_num
		# Writes Tensorboard summaries to disk
		self.summary_writer = None
		with tf.variable_scope(scope):
		# Build the graph
			self.build_model()
			if summaries_dir:
				summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
				if not os.path.exists(summary_dir):
					os.makedirs(summary_dir)
				self.summary_writer = tf.summary.FileWriter(summary_dir)


	def build_model(self):
		self.x = tf.placeholder(shape = [None, 2*self.question_num + self.people_num + 3], dtype = tf.float32)
		# The TD target value
		self.y = tf.placeholder(shape = [None], dtype = tf.float32)
		# Integer id of which actions was selected
		self.actions = tf.placeholder(shape = [None], dtype = tf.int32)

		# MLP1
		batch_size = tf.shape(self.x)[0]
		hidden11 = tf.contrib.layers.fully_connected(self.x, 128)
		hidden12 = tf.contrib.layers.fully_connected(hidden11, 64)
		self.predictions1 = tf.contrib.layers.fully_connected(hidden12, self.actions_num1)
		# Get the predictions for the chosen actions only
		gather_indices1 = tf.range(batch_size)*tf.shape(self.predictions1)[1] + self.actions
		self.action_predictions1 = tf.gather(tf.reshape(self.predictions1, [-1]), gather_indices1)
		# Calculate the loss
		self.loss1 = tf.reduce_mean(tf.squared_difference(self.y, self.action_predictions1))
		# Optimizer Parameters from original paper
		self.optimizer1 = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op1 = self.optimizer1.minimize(self.loss1, global_step=tf.contrib.framework.get_global_step())

		# MLP2
		hidden21 = tf.contrib.layers.fully_connected(self.x, 128)
		hidden22 = tf.contrib.layers.fully_connected(hidden21, 64)
		self.predictions2= tf.contrib.layers.fully_connected(hidden22, self.actions_num2)
		# Get the predictions for the chosen actions only
		gather_indices2 = tf.range(batch_size)*tf.shape(self.predictions2)[1] + self.actions
		self.action_predictions2 = tf.gather(tf.reshape(self.predictions2, [-1]), gather_indices2)
		# Calculate the loss
		self.loss2 = tf.reduce_mean(tf.squared_difference(self.y, self.action_predictions2))
		# Optimizer Parameters from original paper
		self.optimizer2 = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op2 = self.optimizer2.minimize(self.loss2, global_step=tf.contrib.framework.get_global_step())

		# Summaries for Tensorboard
		self.summaries1 = tf.summary.merge([
			tf.summary.scalar("loss", self.loss1),
			tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions1))
		])
		self.summaries2 = tf.summary.merge([
			tf.summary.scalar("loss", self.loss2),
			tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions2))
		])

	def predict_q_values1(self, sess, state):
		"""
		Predicts action values.

		Returns:
		  Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
		  action values.
		"""
		return sess.run(self.predictions1, { self.x: state })
	def predict_q_values2(self, sess, state):
		"""
		Predicts action values.

		Returns:
		  Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
		  action values.
		"""
		return sess.run(self.predictions2, { self.x: state })

	def state_to_action1(self, sess, state, epsilon):
		q_values = self.predict_q_values1(sess, state)
		actions_prob = np.ones(self.actions_num1, dtype = float) * epsilon/self.actions_num1
		best_action = np.argmax(q_values)
		actions_prob[best_action] += (1.0 - epsilon)
		action_id = np.random.choice(np.arange(len(actions_prob)), p=actions_prob)
		return action_id
		# 	for question_no, answer in zip(state['qa_pair'][0], state['qa_pair'][1]):
		# 	self.database[state['people'], question_no] = answer
		# 	return None

	def state_to_action2(self, sess, state, epsilon):
		q_values = self.predict_q_values2(sess, state)
		actions_prob = np.ones(self.actions_num2, dtype = float) * epsilon/self.actions_num2
		best_action = np.argmax(q_values)
		actions_prob[best_action] += (1.0 - epsilon)
		action_id = np.random.choice(np.arange(len(actions_prob)), p=actions_prob)
		return action_id
		# 	for question_no, answer in zip(state['qa_pair'][0], state['qa_pair'][1]):
		# 	self.database[state['people'], question_no] = answer
		# 	return None

	def update1(self, sess, s, a, y):
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
		feed_dict = { self.x: s, self.y: y, self.actions: a}
		summaries, global_step, _, loss = sess.run(
			[self.summaries1, tf.contrib.framework.get_global_step(), self.train_op1, self.loss1],
			feed_dict)
		if self.summary_writer:
			self.summary_writer.add_summary(summaries, global_step)
		return loss

	def update2(self, sess, s, a, y):
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
		feed_dict = { self.x: s, self.y: y, self.actions: a}
		summaries, global_step, _, loss = sess.run(
			[self.summaries2, tf.contrib.framework.get_global_step(), self.train_op2, self.loss2],
			feed_dict)
		if self.summary_writer:
			self.summary_writer.add_summary(summaries, global_step)
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


