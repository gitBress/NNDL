import dill
import numpy as np
import agent
import environment

episodes = 3000         # number of training episodes
episode_length = 50     # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [8, 8]           # objective point
discount = 0.9          # exponential discount factor
softmax = True         # set to true to use Softmax policy
sarsa = True           # set to true to use the Sarsa algorithm

# alpha and epsilon profile
if (sarsa):
    if (softmax):
        alpha = np.linspace(0.9,0.2,episodes)
        epsilon = np.linspace(0.7, 0.001,episodes)
    else:
        alpha = np.ones(episodes) * 0.2
        epsilon = np.linspace(0.7, 0.001,episodes)
else:
    alpha = np.linspace(0.8,0.1,episodes)
    epsilon = np.linspace(0.7, 0.001,episodes)
    

# initialize the agent
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

# Test for 4 fixed final positions
fixed_position = [[1,1], [1,9], [2,7], [9,2]];

track_moves = []
track_start = []
counter = 0
# perform the training
for index in range(0, episodes):
    # start from a random state (but avoid barrier)
    barrier_x = [0,1,2,3,4,6,7,8,9]
    barrier_y = [4,5]
    while(True):
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
        if (not(initial[0] in barrier_y and initial[1] in barrier_x)):
            break
    # common start of last positions to test the result
    if(index > (episodes-len(fixed_position)-1)):
        initial = fixed_position[counter]
        counter = counter + 1
    # initialize environment
    state = initial
    env = environment.Environment(x, y, state, goal)
    reward = 0
    # run episode
    track_moves_tmp = []
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
        result = env.move(action)
        track_moves_tmp.append(result[0]);
        # Q-learning update
        next_index = result[0][0] * y + result[0][1]
        learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
        # update state and reward
        reward += result[1]
        state = result[0]
    reward /= episode_length
    track_moves.append(track_moves_tmp)
    track_start.append(initial)
    print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial) 
    
    # periodically save the agent
    if ((index + 1) % 10 == 0):
        with open('agent.obj', 'wb') as agent_file:
            dill.dump(agent, agent_file)


#%% PRINT MOVES

for k in range(len(fixed_position)):
    print('Testing position #', k+1)
    treasure_map = track_moves[len(track_moves)-len(fixed_position)+(k)]
    checkerboard = np.zeros((x,y), dtype=int)
    guard = True
    for tmp_position in treasure_map:
        checkerboard[tmp_position[0],tmp_position[1]] += 1
        if (tmp_position[0]==8 and tmp_position[1]==8):
            break
    
    start_tmp = track_start[len(track_start)-len(fixed_position)+(k)]
    checkerboard[start_tmp[0],start_tmp[1]] = 2
    checkerboard[8,8] = 3
    for i in range(10):
        print(checkerboard[i])
    print("#####")
    