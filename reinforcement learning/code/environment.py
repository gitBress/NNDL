import numpy as np

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }
    
    def __init__(self, x, y, initial, goal):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
    
    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = 0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            reward = -1
        else:
            self.state = next_state
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        # Add a constriction in the environment: for line 4 and 5, it can only go through positions [4,4],[4,5],[5,4],[5,5]
        barrier_x = [0,1,2,3,4,6,7,8,9]
        barrier_y = [4,5]
        if(state[0] in barrier_y and state[1] in barrier_x):
            out = out + 1
        return out > 0
    
