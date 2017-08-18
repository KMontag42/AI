from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

# # setup xrange
# try:
#     import xrange
# except ImportError:
#     import range as xrange

env = gym.make('CartPole-v0')

################################
####    SETUP THE AGENT     ####
################################

gamma = 0.99

def discount_rewards(r):
    """ r: 1D array of floats, compute discounted reward """
    discounted_reward = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0,r.size)):
        running_add = running_add * gamma + r[t]
        discounted_reward[t] = running_add

    return discounted_reward

class agent():
    def __init__(self, learning_rate, s_size, actions_size, h_size):
        # create placeholder Tensor for the state
        self.state = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        # create hidden first layer to then be translated into an output
        hidden = slim.fully_connected(self.state, h_size, activation_fn=tf.nn.relu, biases_initializer=None)
        # create output network using our hidden network as input, and using the size of the action space as bounds
        self.output = slim.fully_connected(hidden, actions_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        # pull the chosen action out of the output network
        self.chosen_action = tf.argmax(self.output, 1)

        # create placeholder Tensor for our rewards
        self.rewards_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        # create placeholder Tensor for our actions
        self.actions_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        # get the indexes from our output and actions
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.actions_holder
        # get our `responsible_outputs` from output and indexes
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # compute our loss, it is negated
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.rewards_holder)

        # setup trainable variables
        tvars = tf.trainable_variables()

        # setup holder array for policy gradients
        self.gradients_holder = []

        # setup dynamic holders in our gradients_holder for each trainable variable
        for index, variable in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(index)+"_holder")
            self.gradients_holder.append(placeholder)

        # set our policy gradients
        self.gradients = tf.gradients(self.loss, tvars)

        # setup optimizer to use our learning rate param
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # use our optimizer to apply policy gradients, and setup update batch operation
        self.update_batch = optimizer.apply_gradients(zip(self.gradients_holder, tvars))

###############################
####  TRAINING THE AGENT   ####
###############################

# clear the TF graph
tf.reset_default_graph()

# setup our agent
my_agent = agent(learning_rate=1e-2, s_size=4, actions_size=2, h_size=8)

# setup variables for training
total_episodes = 5000
episode_steps = 999
update_frequency = 5

# create our TF initializer
init = tf.global_variables_initializer()

# start up our TF session
with tf.Session() as sess:
    # run our initializer at the start of each session
    sess.run(init)

    # setup session variables
    current_episode = 0
    total_reward = []
    total_length = []

    # get our gradient buffer
    gradient_buffer = sess.run(tf.trainable_variables())

    # zero our gradient buffer
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    # start episode loop
    while current_episode < total_episodes:
        # get observation from the environment
        state = env.reset()

        # setup episode variables
        running_reward = 0
        episode_history = []

        # start step loop for episode
        for j in range(episode_steps):
            # get a distribution of actions from our network
            action_distribution = sess.run(my_agent.output, feed_dict={my_agent.state:[state]})
            # pick a random action out of the distribution
            action = np.random.choice(action_distribution[0], p=action_distribution[0])
            # argmax our previous choice
            action = np.argmax(action_distribution == action)

            # take the action!!
            next_state, reward, done, _ = env.step(action)
            # add episode history
            episode_history.append([state, action, reward, next_state])

            # overwrite our current state
            state = next_state
            # add to our running_reward total
            running_reward += reward

            # handle updating the network when the agent is finished running
            if done == True:
                # reassign episode_history to numpy array for easier manipulation
                episode_history = np.array(episode_history)
                # discount our episode history rewards, the [:,2] syntax gets the array of values from the nested arrays at index 2 (in our case the rewards)
                episode_history[:,2] = discount_rewards(episode_history[:,2])

                # generate dictionary to feed into the gradients operation
                gradients_feed_dict = {
                    my_agent.rewards_holder:episode_history[:,2],
                    my_agent.actions_holder:episode_history[:,1],
                    my_agent.state:np.vstack(episode_history[:,0])
                }

                # run the gradient operation
                gradients = sess.run(my_agent.gradients, feed_dict=gradients_feed_dict)
                # add the new gradients to our buffer
                for index,gradient in enumerate(gradients):
                    gradient_buffer[index] += gradient

                # update the network at our defined frequency
                if current_episode % update_frequency == 0 and current_episode != 0:
                    # generate feed dictionary for the update operation
                    update_feed_dict = dict(zip(my_agent.gradients_holder, gradient_buffer))
                    # run the update operation, we don't care about the output
                    _ = sess.run(my_agent.update_batch, feed_dict=update_feed_dict)

                    # re:zero our gradient_buffer
                    for index, gradient in enumerate(gradient_buffer):
                        gradient_buffer[index] = gradient * 0

                # add to our total rewards
                total_reward.append(running_reward)
                # add to the total length of our session
                total_length.append(j)
                break

        # do occasional logging
        if current_episode % 100 == 0:
            print(np.mean(total_reward[-100:]))

        # increment the current episode number
        current_episode += 1
