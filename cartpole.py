#observation is [position of cart, velocity of cart, angle of pole, rotation rate of pole]
#but a good aganet don't need to know this, the weights gives the chosen priority
def evaluate_params_by_sigmoid(env, weight):
    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        env.render()
        weighted_sum = np.dot(observation, weight)
        if weighted_sum >= 0:
            action = 1
        else:
            action = 0

        observation, reward, done, info = env.step(action)
        total_reward+= reward
        if done:
            break
        pass

    return total_reward

# have some question about this: how to choose is what PG does or it si the basic rule
def choose_action(weights, observation):
    weighted_sum = np.dot(weight, observation)
    policy = 1 / (1 + np.exp(-weighted_sum)) #sigmoid 0 at 0.5
    if policy > 0.5:
        action = 1
    else:
        action = 0
    return action, policy

def generate_episodes(env, weights):
    episode = []
    pre_observation = env.reset()
    t = 0
    while 1:
        # env.render()
        action, policy = choose_action(weight,pre_observation)
        observation,reward,done,info = env.step(action)
        #save this episode
        episode.append([pre_observation, action, policy, reward])
        pre_observation = observation

        t += 1
        if done or t > 1000:
            break
    return episode

# input is the observation, output should be the action probability

def monte_carlo_policy_agent(env):
    learning_rate = -0.001
    best_reward = -100

    weight = np.random.rand(4)

    for iter in xrange(1000):
        current_episode = generate_episodes(env,weight)
        for t in range:
            observation, action, pi, reward = current_episode[t]
            weight += learning_rate * (1-policy)* np.transpose(-observation)*reward

    current_reward = evaluate_given_parameter_sigmoid(env, weight)
    print 'Monte-Carlo policy gradient get reward', current_reward

def actor_critic_policy_gradient(env):
    gamma = 1

    p_weight = np.random.rand(4) #policy weights
    v_weight = np.random.rand(4)

    p_learning_rate = -0.0001
    v_learning_rate = -0.0001

    done = True
    for iter in xrange(1000):
        t = 0
        while 1:
            if done:
                print('start new training...')
                print('policy weights' + str(p_weight))
                print('value weights' + str(v_weight))

                pre_observation = env.reset()
                pre_policy, pre_action = choose_action(p_weight,pre_observation)

                pre_phi = pre_observation
                pre_Q = np.dot(v_weight, pre_phi)
            pass

            observation, reward, done, info = env.step(pre_action)

            policy, action = choose_action(p_weight, observation)

            phi = observation
            Q = np.dot(phi, v_weight)

            delta = reward + gamma * Q - pre_Q
            p_weight += p_learning_rate * (1-policy) * pre_Q
            v_weight += v_learning_rate * delta * np.transpose(pre_phi)

            #update
            pre_policy = policy
            pre_observation = observation
            pre_Q = Q
            pre_phi = phi
            pre_action = action

            t+=1
            if done:
                break
                pass
        pass

    cur_reward = evaluate_given_parameter_sigmoid(env, p_weight)
    print('Actor critic policy gradient get reward'+ str(cur_reward))
