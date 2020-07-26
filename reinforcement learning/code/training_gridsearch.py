
import numpy as np
import agent
import environment

episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [8, 8]           # objective point
discount = 0.9          # exponential discount factor
softmax = True         # set to true to use Softmax policy
sarsa = True           # set to true to use the Sarsa algorithm


#%% K-fold cross-validation

# Parameters grid
params = {
    'alpha_fixed': [True, False],
    'alpha_start':[0.9, 0.8],
    'alpha_end':[1e-1, 2e-1],
    'alpha':[0.2, 0.4, 0.6, 0.8],
    'epsilon_start': [0.9, 0.8, 0.7],
    'epsilon_end':[1e-3, 1e-2],
    'episodes': [3000],
}

final_reward = []
opt_reward = -1
opt_alpha_fixed = False
opt_alpha = 0
opt_alpha_start = 0
opt_alpha_end = 0
opt_epsilon_start = 0
opt_epsilon_end = 0
opt_episodes = 0

counter = 0
# Cross-validation
for episodes in params['episodes']:
    for epsilon_start in params['epsilon_start']:
        for epsilon_end in params['epsilon_end']:
            for alpha_fixed in params['alpha_fixed']:
                if (alpha_fixed):
                    for alpha in params['alpha']:
                        # alpha and epsilon profile
                        alpha = np.ones(episodes) * alpha
                        epsilon = np.linspace(epsilon_start, epsilon_end,episodes)
                        
                        # initialize the agent
                        learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
                        # perform the training
                        rewards = []
                        for index in range(0, episodes):
                            # start from a random state (but avoid barrier and mountain)
                            barrier_x = [0,1,2,3,4,6,7,8,9]
                            barrier_y = [4,5]
                            while(True):
                                initial = [np.random.randint(0, x), np.random.randint(0, y)]
                                if (not(initial[0] in barrier_y and initial[1] in barrier_x)):
                                    break
                            # initialize environment
                            state = initial
                            env = environment.Environment(x, y, state, goal)
                            reward = 0
                            # run episode
                            for step in range(0, episode_length):
                                # find state index
                                state_index = state[0] * y + state[1]
                                # choose an action
                                action = learner.select_action(state_index, epsilon[index])
                                # the agent moves in the environment
                                result = env.move(action)
                                # Q-learning update
                                next_index = result[0][0] * y + result[0][1]
                                learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
                                # update state and reward
                                reward += result[1]
                                state = result[0]
                            reward /= episode_length
                            rewards.append(reward)
                        print('Set ', counter + 1, ': the agent has obtained an average reward of ', np.mean(rewards)) 
                        counter = counter + 1 
                        if (np.mean(rewards)>=opt_reward):
                            opt_reward = np.mean(rewards)
                            opt_alpha_fixed = True
                            opt_alpha = alpha
                            opt_epsilon_start = epsilon_start
                            opt_epsilon_end = epsilon_end
                            opt_episodes = episodes
                else:
                    for alpha_start in params['alpha_start']:
                        for alpha_end in params['alpha_end']:
                            # alpha and epsilon profile
                            alpha = np.linspace(alpha_start,alpha_end,episodes)
                            epsilon = np.linspace(epsilon_start,epsilon_end,episodes)
                            
                            # initialize the agent
                            learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
                            # perform the training
                            rewards = []
                            for index in range(0, episodes):
                                # start from a random state (but avoid barrier)
                                barrier_x = [0,1,2,3,4,6,7,8,9]
                                barrier_y = [4,5]
                                while(True):
                                    initial = [np.random.randint(0, x), np.random.randint(0, y)]
                                    if (not(initial[0] in barrier_y and initial[1] in barrier_x)):
                                        break
                                # initialize environment
                                state = initial
                                env = environment.Environment(x, y, state, goal)
                                reward = 0
                                # run episode
                                for step in range(0, episode_length):
                                    # find state index
                                    state_index = state[0] * y + state[1]
                                    # choose an action
                                    action = learner.select_action(state_index, epsilon[index])
                                    # the agent moves in the environment
                                    result = env.move(action)
                                    # Q-learning update
                                    next_index = result[0][0] * y + result[0][1]
                                    learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
                                    # update state and reward
                                    reward += result[1]
                                    state = result[0]
                                reward /= episode_length
                                rewards.append(reward)
                            print('Set ', counter + 1, ': the agent has obtained an average reward of ', np.mean(rewards)) 
                            counter = counter + 1
                            if (np.mean(rewards)>=opt_reward):
                                opt_reward = np.mean(rewards)
                                opt_alpha_fixed = False
                                opt_alpha_start = alpha_start
                                opt_alpha_end = alpha_end
                                opt_epsilon_start = epsilon_start
                                opt_epsilon_end = epsilon_end
                                opt_episodes = episodes
                        
# Optimal parameters grid
opt_params = {
    'opt_alpha_fixed': [opt_alpha_fixed],
    'opt_alpha_start':[opt_alpha_start],
    'opt_alpha_end':[opt_alpha_end],
    'opt_alpha':[opt_alpha],
    'opt_epsilon_start': [opt_epsilon_start],
    'opt_epsilon_end':[opt_epsilon_end],
    'opt_episodes': [opt_episodes],
}

print('OPTIMAL SET OF PARAMETERS: ' + str(opt_params))