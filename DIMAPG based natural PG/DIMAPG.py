# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:10:33 2018

@author: Xzw

E-mail: diligencexu@gmail.com
"""

import tensorflow as tf
import numpy as np

EPS = 1e-8

class DIMAPG(object):
    def __init__(self, sess, feature_dim, action_dim, N, batch_size=32, alpha=1e-3, epsilon=1e-3, gamma=0.9, hidden_units=64):
        self.sess = sess
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.N = N
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        
        self.s_buf, self.a_buf, self.r_buf = [], [], []
        
        self.S = tf.placeholder(tf.float32, [None, self.feature_dim])
        self.A = tf.placeholder(tf.float32, [None, self.action_dim])
        self.R = tf.placeholder(tf.float32, [None, 1])
        
        self.CP_pi, self.CP_logp, self.CP_logp_pi = self._build_net('Central_policy')
        self.DP_pi, self.DP_logp, self.DP_logp_pi = self._build_net('Decentral_policy')
        
        self.cp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Central_policy_PG')
        self.dp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Decentral_policy_PG')
        
        self.initialize = [tf.assign(d,c) for d, c in zip(self.dp_params, self.cp_params)]
        
        self.DP_loss = -self.R * self.DP_logp
        
        self.DP_train = tf.train.AdamOptimizer(self.alpha).minimize(self.DP_loss)
        
        self.DP_train_param = tf.trainable_variables('Decentral_policy_PG')
        self.CP_trainer = tf.train.AdamOptimizer(self.epsilon)
        self.gradient_all = self.CP_trainer.compute_gradients(self.DP_loss, self.DP_train_param)
        self.grads_vars = [v for (g, v) in self.gradient_all if g is not None]
        self.gradient = self.CP_trainer.compute_gradients(self.DP_loss, self.grads_vars)
        self.grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self.gradient]
        self.v = [v for (g, v) in self.gradient]
        self.CP_train = self.CP_trainer.apply_gradients(self.grads_holder)
        self.tmp_grads = []
        
    def _build_net(self,name):
        with tf.variable_scope(name + '_PG'):
            l1 = tf.layers.dense(inputs=self.S,
                                 units=self.hidden_units,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 activation=tf.nn.relu)
            
            l2 = tf.layers.dense(inputs=l1,
                                 units=self.action_dim,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 activation=tf.nn.relu)
            
            mu = tf.layers.dense(inputs=l2,
                                 units=self.action_dim,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 activation=None)
            
            log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(self.action_dim, dtype=np.float32),trainable=True)
            
            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp = self.gaussian_likelihood(self.A, mu, log_std)
            logp_pi = self.gaussian_likelihood(pi, mu, log_std)
            
            return pi, logp, logp_pi
        
    
    def gaussian_likelihood(self, a, mu, log_std):
        pre_sum = -0.5 * (((a-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
    
    
    def discount_reward(self, n):
        cumulative_reward = np.zeros_like(np.array(self.r_buf[n]))
        running_add = 0
        for i in reversed(range(len(self.r_buf[n]))):
            running_add = running_add * self.gamma + np.array(self.r_buf[n][i])
            cumulative_reward[i] = running_add
        cumulative_reward = cumulative_reward.sum(axis=1)
        cumulative_reward = (cumulative_reward - np.mean(cumulative_reward)) / np.std(cumulative_reward + EPS)
        return cumulative_reward
    
    
    def get_sample(self, i, batch_size):
        count = len(self.s_buf[i])
        sample_index = np.random.choice(np.arange(count), batch_size)
        cum_reward = self.discount_reward(i)
        return np.array(self.s_buf[i])[sample_index,0], np.squeeze(np.array(self.a_buf[i])[sample_index,0]), np.array([cum_reward[sample_index]]).T
    
    
    def DP_learn(self):
        s, a, r = self.get_sample(-1, self.batch_size)
        loss = self.sess.run(self.DP_loss, feed_dict={self.S: s, self.A: a, self.R: r})        
        grad, _ = self.sess.run([self.gradient, self.DP_train], feed_dict={self.S: s, self.A: a, self.R: r})
        self.tmp_grads.append(grad)
        return loss, grad
                
        
    def CP_learn(self):
        grads_sum = {}
        for i in range(len(self.grads_holder)):
            k = self.grads_holder[i][0]
            grads_sum[k] = sum([g[i][0] for g in self.tmp_grads])
        self.sess.run(self.CP_train, grads_sum)
        self.tmp_grads = []
    
    def store_trajectory(self, s, a, r, new):
        if new:
            self.a_buf.append([])
            self.s_buf.append([])
            self.r_buf.append([])
        self.a_buf[-1].append(a)
        self.s_buf[-1].append(s)
        self.r_buf[-1].append(r)
        
        
    def choose_action(self, s):
        action = [self.sess.run(self.DP_pi, feed_dict={self.S: np.array(s[0])[None,:]})]
        for i in range(self.N-1):
            action.append(self.sess.run(self.DP_pi, feed_dict={self.S: np.array(s[i+1])[None,:]}))
        return action
    
    
    def initial(self):
        self.sess.run(self.initialize)
        
    
    def clean_trajectory(self):
        self.s_buf, self.a_buf, self.r_buf = [], [], []
    
