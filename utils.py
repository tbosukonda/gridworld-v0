import matplotlib.pyplot as plt 
import numpy as np 
import random

class MDP:

    def __init__(self, num_rows, num_cols, terminals, rewards, gamma, noise = 0.0):

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.action_space = [0,1,2,3] # 0: down, 1 : up, 2:left, 3:right
        self.terminals = terminals
        self.rewards = rewards 
        self.gamma = gamma
        self.noise = noise
        self.start_state = (0,0)
        self.states = [(r,c) for r in range(self.num_rows) for c in range(self.num_cols)]
        #self.done = False
        self.maxsteps = 10
        self.transition_mat = np.zeros((len(self.states), len(self.action_space), len(self.states)))
        self.transition_matrix()

    def transition_matrix(self):

        noise = self.noise

        #for going down i.e., action 0
        for s in self.states:
            r = s[0]
            c = s[1]
            cs = (self.num_rows*r) + c
            
            if r == 0:
                self.transition_mat[cs][0][cs]+= 1.0 - (3*noise)
            else:
                self.transition_mat[cs][0][cs-3]+= 1.0 - (3*noise)
            
            #current state is leftmost cell and going left
            if c == 0:
                self.transition_mat[cs][2][cs]+= noise
            else:
                self.transition_mat[cs][2][cs-1]+= noise 

            #current state is rightmost cell and going right
            if c == 2:
                self.transition_mat[cs][3][cs]+= noise
            else:
                self.transition_mat[cs][3][cs+1]+= noise 
            
            #current state is topmost cell and going up
            if r==2:
                self.transition_mat[cs][1][cs]+= noise
            else:
                self.transition_mat[cs][1][cs+3]+= noise

        

        #for going up i.e., action 1
        for s in self.states:
            r = s[0]
            c = s[1]
            cs = (self.num_rows*r) + c
            
            if r == 2:
                self.transition_mat[cs][1][cs]+= 1.0 - (3*noise)
            else:
                self.transition_mat[cs][1][cs+3]+= 1.0 - (3*noise)
            
            #current state is leftmost cell and going left
            if c == 0:
                self.transition_mat[cs][2][cs]+= noise
            else:
                self.transition_mat[cs][2][cs-1]+= noise 

            #current state is rightmost cell and going right
            if c == 2:
                self.transition_mat[cs][3][cs]+= noise
            else:
                self.transition_mat[cs][3][cs+1]+= noise 
            
            #current state is bottommost cell and going down
            if r==0:
                self.transition_mat[cs][0][cs]+= noise
            else:
                self.transition_mat[cs][0][cs-3]+= noise 
        
        

        #for going left i.e., action 2
        for s in self.states:
            r = s[0]
            c = s[1]
            cs = (self.num_rows*r) + c
            
            if c == 0:
                self.transition_mat[cs][2][cs]+= 1.0 - (3*noise)
            else:
                self.transition_mat[cs][2][cs-1]+= 1.0 - (3*noise)
            
            #current state is leftmost cell and going down
            if r == 0:
                self.transition_mat[cs][0][cs]+= noise
            else:
                self.transition_mat[cs][0][cs-3]+= noise 

            #current state is rightmost cell and going right
            if c == 2:
                self.transition_mat[cs][3][cs]+= noise
            else:
                self.transition_mat[cs][3][cs+1]+= noise 
            
            #current state is topmost cell and going up
            if r==2:
                self.transition_mat[cs][1][cs]+= noise
            else:
                self.transition_mat[cs][3][cs+3]+= noise

        #for going right i.e., action 3
        for s in self.states:
            r = s[0]
            c = s[1]
            cs = (self.num_rows*r) + c

            if c == 2:
                self.transition_mat[cs][3][cs]+= 1.0 - (3*noise)
            else:
                self.transition_mat[cs][3][cs+1]+= 1.0 - (3*noise)
            
            #current state is leftmost cell and going left
            if c == 0:
                self.transition_mat[cs][2][cs]+= noise
            else:
                self.transition_mat[cs][2][cs-1]+= noise 

            #current state is rightmost cell and going down
            if r == 0:
                self.transition_mat[cs][0][cs]+= noise
            else:
                self.transition_mat[cs][0][cs-3]+= noise 
            
            #current state is topmost cell and going up
            if r==2:
                self.transition_mat[cs][1][cs]+= noise
            else:
                self.transition_mat[cs][3][cs+3]+= noise

        #checking for terminal state
        for s in self.states:
            r = s[0]
            c = s[1]
            cs = (self.num_rows*r) + c
            if s in self.terminals:
                for a in self.action_space:
                    for s in self.states:
                        ns = (self.num_rows*s[0]) + s[1]
                        self.transition_mat[cs][a][ns] = 0.0


                    
    def get_action(self, curr_state, next_state):
        # returns the list of possible actions that can be taken from the current state to reach the next state
        # the current and next state are in the form a tuple (row, col)

        cs = (self.num_rows * curr_state[0]) + curr_state[1]
        ns = (self.num_rows * next_state[0]) + next_state[1]
        possible_actions = []

        for action in self.action_space:
            if self.transition_mat[cs][action][ns] != 0:
                possible_actions.append(action)

        return possible_actions
        

    def next_state(self, curr_state, action):
        #return the next state given the current state and action
        #current state is in the form of (row, col)
        #action can is an integer with the mapping:  0: down, 1 : up, 2:left, 3:right
        
        ns = int(np.where(self.transition_mat[curr_state][action] == 1.0)[0])
        nr = ns//self.num_rows
        nc = ns%self.num_rows

        return (nr, nc)


    def step(self, curr_state, action, timestep, done):
        #returns the next state and the reward associated for taking the action
        #current state is a (row, col) tuple

        cs = (self.num_rows * curr_state[0]) + curr_state[1]
        ns = self.next_state(cs, action)
        timestep+= 1
        reward = self.rewards[ns]
        if ns in self.terminals or timestep >= self.maxsteps:
            done = True
        
        plot_gridworld(ns)

        return ns, reward, done, timestep

    def random_step(self, curr_state, timestep, done):
        #randomly picks an action from the action space
        #returns the next state and the associalted reward

        action = random.sample(self.action_space, 1)
        # print("Action is: ", *action)
        return self.step(curr_state, *action, timestep, done)
        

def plot_gridworld(curr_state):
    #prints a user friendly representation of the griworld

    grid = [['', '', ''], ['', '', '']]
    fig, ax = plt.subplots()

    plt.xticks(np.arange(0, 3.1, 1))
    plt.yticks(np.arange(0, 3.1, 1))
    plt.axis('equal')

    # plt.axis('off')
    for i in range(3):
        for j in range(3):

            # Draw the cell rectangle
            if (i,j) == curr_state:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black')
                 # Add the cell value as text
                ax.text(j + 0.5, i + 0.5, "*", ha='center', va='center')

            elif i==2 and j==2:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='green', edgecolor='black')
                ax.text(j + 0.5, i + 0.5, "G", ha='center', va='center')

            else:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black')

            ax.add_patch(rect)

    plt.show()

def human_demonstration(env):
    #takes a demonstration from the human

    done = False
    curr_state = env.start_state
    reward_history = []
    state_history = []
    timestep = 0

    while not done:
        state_history.append(curr_state)
        action = int(input("Enter action: "))
        ns, reward, done, timestep = env.step(curr_state, action, timestep, done)
        reward_history.append(reward)
        curr_state = ns
    print("Your trajectory is: ", *state_history)
    print("Your final reward for the demonstration is ", np.sum(reward_history))
    return state_history, reward_history


def random_demonstration(env):
    #randomly generates a demonstration

    done = False
    curr_state = env.start_state
    reward_history = []
    state_history = []
    timestep = 0

    while not done:
        state_history.append(curr_state)
        ns, reward, done, timestep = env.random_step(curr_state, timestep, done)
        reward_history.append(reward)
        curr_state = ns
    print("Random trajectory is: ", *state_history)
    print("Final reward for the trajectory is ", np.sum(reward_history))
    return state_history, reward_history


if __name__ == "__main__":

    #define rewards dictionary
    rewards = {(0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1, (2,0): -1, (2,1): -1, (2,2): 5}
    env = MDP(3, 3, [(2, 2)], rewards, 0.9)

    # print(env.transition_mat)

    print("The GREEN cell is the GOAL or TARGET cell.\nYour CURRENT STATE is represented as a BLUE cell with a star")
    plot_gridworld((0,0))
    print("Use the following to move in the respective directions:\nEnter 0 to move DOWN a cell\nEnter 1 to move UP a cell\nEnter 2 to move LEFT a cell\nEnter 3 to move RIGHT a cell")
    demonstrations = []
    rewards = []

    # taking human demonstration
    demonstration, reward_history = human_demonstration(env)
    demonstrations.append(demonstration)
    rewards.append(reward_history)

    #generating a random demonstration
    demonstration, reward_history = random_demonstration(env)
    demonstrations.append(demonstration)
    rewards.append(reward_history)


    