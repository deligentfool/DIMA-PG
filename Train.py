# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:55:12 2018

@author: Xzw

E-mail: diligencexu@gmail.com
"""

import make_env
from DIMAPG import DIMAPG
import tensorflow as tf
import numpy as np

MAX_EPISODE = 100000000
SINGLE_EPISODE = 150
NOISE = 3
N = 3
RECORD_SIZE = SINGLE_EPISODE * N * 2
BATCH_SIZE = 32

if __name__ == '__main__':
    env = make_env.make_env('simple_tag')
    
    feature_dim = 16
    action_dim = 2
    
    sess = tf.Session()
        
    RL = DIMAPG(sess, feature_dim, action_dim, N, BATCH_SIZE)
    
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    RL.initial()
    

    reward_info = np.zeros(RECORD_SIZE)
    observation = env.reset()
    flag = True
    tag_play = True
    for episode in range(1, MAX_EPISODE+1):
        action_adv = RL.choose_action(observation)

        action = [[0, a[0,0] + np.random.randn() * NOISE, 0, a[0,1] + np.random.randn() * NOISE, 0] for a in action_adv]
        action.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
        
        observation_, reward, done, info = env.step(action)

        reward_info[(episode - 1) % RECORD_SIZE] = reward[0] + reward[1] + reward[2]

        RL.store_trajectory(observation[: -1], action_adv, reward[: -1], flag)
        flag = False
        if 'running_reward' not in globals():
            running_reward = reward_info.sum()
        else:
            running_reward = 1e-5 * reward_info.sum() + (1 - 1e-5) * running_reward
        
        if episode >= RECORD_SIZE * 100:
            if tag_play:
                print('start learning!!')
                tag_play = False
            if episode % SINGLE_EPISODE == 0:
                RL.DP_learn()
        
            if episode % (SINGLE_EPISODE * N * 2) == 0:
                RL.CP_learn()
                flag = True
                RL.clean_trajectory()

            
            
            if episode % (SINGLE_EPISODE * 2) == 0:
                RL.initial()
                observation = env.reset()
                flag = True
            
            if episode % (SINGLE_EPISODE * N * 2 * 100) == 0:
                saver.save(sess, './simple_tag_ma_weight/' + str(int(episode/(SINGLE_EPISODE*N*2))) + '.cptk')

        if episode % (SINGLE_EPISODE * N * 2) == 0:
            print('episode :{} running reward :{:.3f}'.format(episode,running_reward))
            