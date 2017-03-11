#Notes: We are using the Box2D environment (Along with the cartpole_env.py )
# The reward is just a number. Not sure what is the dimension of the observation, but can check that easily by
# printing out (have feeling its 4X1)

def sigma_policy(obs):
    
    
    with tf.variable_scope("sigmaPolicy"):
    	out = obs
        # original architecture
        out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
    
        out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
        sigma = layers.fully_connected(out, num_outputs=1, activation_fn=None)

        return sigma

def mu_policy(obs):
    
    with tf.variable_scope("muPolicy"):
        out = obs
        out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
    
        out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
        mu = layers.fully_connected(out, num_outputs=1, activation_fn=None)

        return mu







