import gym
import itertools
import numpy as np 
import random
import sys
import tensorflow as tf 
from collections import deque, namedtuple

class StateProcessor():
	def __init__(self):
		with tf.variable_scope('state_processor'):
			self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
    	return sess.run(self.output, {self.input_state:state})

class Estimator():
	"""
	Q-value Estimator neural network.
	This network is used for both the Q-Network and the Target Network
	"""
	def __init__(self, scope = 'estimator', summaries_dir = None):
		self.scope = scope
		self.summary_writer = None
		with tf.variable_scope(scope):
			self.build_model()
			if summaries_dir:
				summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def build_model(self):
    	self.x = tf.placeholder(shape = [None, d], dtype = tf.float32)
		# The TD target value
		self.y = tf.placeholder(shape = [None], dtype = tf.float32)
		# Integer id of which actions was selected
		self.actions = tf.placeholder(shape = [None], dtype = tf.in32)
		batch_size = tf.shape(self.x)[0]

		hidden1 = tf.contrib.layers.fully_connected(self.x, 128)
		hidden2 = tf.contrib.layers.fully_connected(hidden1, 64)
		self.predictions = tf.contrib.layers.fully_connected(hidden2, len(actions))
		# Get the predictions for the chosen actions only
		gather_indices = tf.range(batch_size)*tf.shape(self.predictions)[1] + self.actions
		self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
		# Calculate the loss
		self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.action_predictions))
		# Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # summaries for tensorboard
        self.summaries = tf.summary.merge([
        	tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
    	return sess.run(self.predictions, {self.x: s})

    def update(self, sess, s, a, y):
    	"""
    	Updates the estimator towards the given targets
    	"""
    	feed_dict = {self.x: s, self.y: y, self.actons:a}
    	summaries, global_step, _, loss = sess.run(
    		[self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
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

def make_epsilon_greedy_policy(estimator, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon

	Returns:
		A function that takes the (sess, observations, epsilon) as an argument and returns
		the probability for each action in the form of a numpy array of length nA
	"""
	def policy_fn(sess, observation, epsilon):
		A = np.ones(nA, dtype = float) * epsion / nA
		q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn

# def deep_q_learning(sess, 
# 	 				env, 
# 	 				q_estimator,
# 	 				target_estimator, 
# 	 				state_processor,
# 	 				num_episodes, 
# 	 				experiment_dir, 
# 	 				replay_memory = 500000, 
# 	 				replay_memory_init_size = 50000,
# 	 				update_target_estimator_every = 10000,
# 	 				discount_factor = 0.99,
# 	 				epsilon_start = 1.0,
# 	 				epsilon_end = 0.1,
# 	 				epsilon_decay_steps = 500000,
# 	 				batch_size = 32, 
# 	 				record_video_every = 50):
# 	Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
# 	replay_memory = []
# 	estimator_copy = ModelParameterCopier(q_estimator, target_estimator)
# 	# Keeps track of useful statistics
# 	stats = 













# For Testing
tf.reset_default_graph()
global_step = tf.Variable(0, name = 'global_step', trainable = False)
e = Estimator(scope = 'test')
sp = StateProcessor()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Example observation batch
	observation = env.reset()

	observation_p = sp.process(sess, observation)