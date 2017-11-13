import tensorflow as tf
import math
import numpy as np
import cPickle as pickle
import gym
import matplotlib.pyplot as plt
%matplotlib inline

env = gym.make('CartPole-v0') #load environment

gamma = 0.99
batch_size = 5 # every how many episodes to do a param update?
xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
learning_rate = 1e-2 

state_size = 4 
action_size = 1
hidden_size = 10

init = tf.initialize_all_variables()

def discount_rewards(r):
    discount_rate = np.zero_like(r)
    add_on = 0
    for t in reversed(range(0,r.size)):
        add_on = add_on* gamma + r[t]
        discount_rate[t] = add_on
    return discount_rate


# self.state_in = tf.placeholder(shape = [None,state_size],dtype = tf.float32)
# hidden = slim.fully_connected(self.state_in, hidden_size, biases_initializer = None, activation_fn = tf.nn.relu)
# self.output = slim.fully_connected(hidden, action_size, activation_fn = tf.nn.softmax, biases_initializer = None)

#now get the current reward and action to compute the policy gradient
self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
self.observation = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

W1 = tf.get_variable("W1", shape=[state_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[hidden_size, action_size],initializer=tf.contrib.layers.xavier_initializer())
output = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(output)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="output")
advantages = tf.placeholder(tf.float32,name="discounted reward")

#input_y is 0 or 1 the likelihhod is probability or 1- probability
log_likelihood = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))

loss = -tf.reduce_mean(loglik * advantages) 
newGrads = tf.gradients(loss,tvars)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = optimizer.apply_gradients(zip(batchGrad,tvars)) #apply after some episodes

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        # Rendering the environment slows things down, 
        # so let's only look at it once our agent is doing a good job.
        if reward_sum/batch_size > 100 or rendering == True : 
            env.render()
            rendering = True
    
        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,state_size])
        tfprob = sess.run(probability,feed_dict={observations: x})
        
        # Run the policy network and get an action to take. 
        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0 #if tfprob is bigger than 0.5, then action = 1
        xs.append(x) # observation
        y = 1 if action == 0 else 0 # a "fake label"
        ys.append(y)
        
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0: 
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('Average reward for episode' + str(reward_sum/batch_size) + 'Total average reward' + str(running_reward/batch_size)) 
                if reward_sum/batch_size > 200: 
                    print "Task solved in",episode_number,'episodes!'
                    break
                reward_sum = 0  

            observation = env.reset()    

print(str(episode_number) + 'Episodes completed.')