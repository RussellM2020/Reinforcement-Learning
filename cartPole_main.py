
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from CartPole.py import sigma_policy
from CartPole.py import mu_policy

K =100
#K is total number of experiments
N = 100
#N is number of trajectories in one experiment
T = 100
#T is number of timesteps in one trajectory
#gamma = 0.99



env = CartpoleEnv() 
inputShape = env.observation_space.shape

def collection():
	observation_History=np.empty(shape = inputShape, dtype = float32)
    action_History = np.empty(shape=1, dtype = float32)
    rewardSum_History=np.empty(shape=1, dtype = float32)

	for trajectory in N:
	#For each of the N iterations,

		observation = env.reset()
		for timestep in T:

			mu = mu_policy(observation)
			sigma = sigma_policy(observation)
		    action = tf.random_normal(shape = 1, mean=mu, stddev=sigma, dtype=tf.float32) 

		    obervation_History.append(observation)
		    action_History.append(action)
		
		    next_observation, new_reward, _ = env.step(action)
		    new_rewardSum = sum(rewardSum_History)+new_reward
		    rewardSum_History.append(new_rewardSum)

	return observation_History, action_History, rewardSum_History


x_ph = tf.placeholder(tf.float32, shape = [None] + list(inputShape))
a_ph = tf.placeholder(tf.float32, shape =[None])
rsum_ph = tf.placeholder(tf.float32, shape=[None])

sigma_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sigmaPolicy')
mu_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='muPolicy')

optimizer = new tf.train.AdamOptimizer()

objective = (-log(sigma_policy(x_ph)) - ((a_ph - mu_policy(x_ph))**2)/(2*(sigma_policy(x_ph)**2))) * rsum_ph
train_sigma = optimizer.minimize(loss = -objective, var_list = sigma_vars)
train_mu = optimizer.minimize(loss = -objective, var_list = mu_vars)


for experiment in K:
	obsH, actH, rewSumH = collection()
	session.run(train_sigma, feed_dict={x_ph: obsH, a_ph:actH, rsum_ph :rewSumH})
	session.run(train_mu, feed_dict={x_ph: obsH, a_ph:actH, rsum_ph :rewSumH})






		



