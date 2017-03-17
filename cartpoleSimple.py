
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize


import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

session = tf.InteractiveSession()


def mupolicy(obs):

    with tf.variable_scope("muPolicy"):
        out =obs
        
       
        out = layers.convolution2d(out, num_outputs=10, kernel_size=3, stride=1, padding = "SAME",activation_fn=tf.nn.relu, trainable= True,
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        out = layers.flatten(out)
    
        out = layers.fully_connected(out, num_outputs=5, activation_fn=tf.nn.relu, trainable = True, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        value = layers.fully_connected(out, num_outputs=1, activation_fn=None, trainable = True, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        
        return value


def sigmapolicy(obs):
    with tf.variable_scope("sigmaPolicy"):
        out = obs
        
        out = layers.convolution2d(out, num_outputs=10, kernel_size=3, stride=1, padding = "SAME",activation_fn=tf.nn.relu, trainable = True, 
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        out = layers.flatten(out)
    
        out = layers.fully_connected(out, num_outputs=5, activation_fn=tf.nn.relu, trainable = True, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        value = layers.fully_connected(out, num_outputs=1, activation_fn=None, trainable = True, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        
        return value


K =100
#K is total number of experiments
N = 30
#N is number of trajectories in one experiment
T = 100
#T is number of timesteps in one trajectory




env = normalize(CartpoleEnv())
inputShape = env.observation_space.shape



def collection():
    rMetric = []
    observation_History=np.empty(shape = (1,1)+inputShape, dtype = float)
    action_History = np.empty(shape=1, dtype = float)
    rewardSum_returnable=np.zeros(shape=1, dtype = float)
    

    for trajectory in range(N):

        observation = env.reset()
        #The observation is reset for the new trajectory
        
        rewardSum_History = np.zeros(shape =1, dtype = float)
        #The sum of rewards needs to be reset for each separate trajectory.
        for timestep in range(T):
 
            mu = session.run(mu_policy, feed_dict = {x_ph: [[observation]]})
            
            log_sigma = session.run(sigma_policy, feed_dict = {x_ph: [[observation]]})
            sigma = np.exp(log_sigma)
            
            observation_History = np.vstack((observation_History,[[observation]]))
            
            # Placeholders take inputs of shape (?,1,4)
            # While collection, each observation is of shape (4,)
            #Thus, we box it twice to give it size (1,1,4)
            
            
            action = np.random.normal(mu, sigma)
            action_History = np.concatenate((action_History,action[0]))
           
            
        
            next_observation, new_reward, _,_ = env.step(action)
            rewardSum_History+=new_reward           
            rewardSum_History= np.concatenate((rewardSum_History, [new_reward]))

            observation = next_observation


        rewardSum_History = np.delete(rewardSum_History,(0))
        rMetric.append(rewardSum_History[-1])
        rewardSum_returnable = np.concatenate((rewardSum_returnable, rewardSum_History))
        #Rewards of a particular trajectory are collected

            
    observation_History = np.delete(observation_History,(0),axis = 0)
    action_History = np.delete(action_History,(0))
    rewardSum_returnable = np.delete(rewardSum_returnable,(0))
    
    return observation_History, action_History, rewardSum_returnable, rMetric


x_ph = tf.placeholder(tf.float32, shape = ([None]+[1]+list(inputShape)))
a_ph = tf.placeholder(tf.float32, shape =[None])
rsum_ph = tf.placeholder(tf.float32, shape=[None])


sigma_policy = sigmapolicy(x_ph)

mu_policy = mupolicy(x_ph) 

sigma_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='sigmaPolicy')
mu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='muPolicy')



optimizer = tf.train.AdamOptimizer()

objective = tf.reduce_mean((-tf.log(tf.abs(sigma_policy)+(10**-3)) - ((a_ph - mu_policy)**2)/(2*(sigma_policy**2))) * rsum_ph)
objective2 = tf.reduce_mean(-tf.log(tf.abs(sigma_policy))+(10**-3))


train_func = optimizer.minimize(loss = -objective, var_list = sigma_vars + mu_vars)
session.run(tf.global_variables_initializer())

for experiment in range(K):
    obsH, actH, rewardsum , rMetric= collection()

    print("At the end of experiment "+ str(experiment) + "real objective is")
    print(session.run(objective, feed_dict={x_ph : obsH, a_ph:actH, rsum_ph: rewardsum}))
    print("At the end of experiment "+ str(experiment) + "sigma policy is")
    print(session.run(objective2, feed_dict={x_ph : obsH, a_ph:actH, rsum_ph: rewardsum}))
    

    
    session.run(train_func, feed_dict={x_ph: obsH, a_ph:actH, rsum_ph :rewardsum})

    






        



