import gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time

import torch
import torch.functional as F
import torch.nn as nn
import json


class AgentNN(nn.Module): #its the Agent network in the ES, and the PolicyNetwork in the PPO 
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)
    
    def load_network(self, network):
        self.fc1.weight = nn.Parameter(torch.from_numpy(network['Layer 1']))
        self.fc2.weight = nn.Parameter(torch.from_numpy(network['Layer 2']))
        self.fc3.weight = nn.Parameter(torch.from_numpy(network['Layer 3']))

        self.fc1.bias = nn.Parameter(torch.from_numpy(network['Bias 1']))
        self.fc2.bias = nn.Parameter(torch.from_numpy(network['Bias 2']))
        self.fc3.bias = nn.Parameter(torch.from_numpy(network['Bias 3']))
    
    def load_network_and_save(self, network, path):
        self.load_network(network)
        torch.save(self.state_dict(), path)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

def glorot_uniform(n_inputs,n_outputs,multiplier=1.0):
    ''' Glorot uniform initialization '''
    glorot = multiplier*np.sqrt(6.0/(n_inputs+n_outputs))
    #Xavier_uniform
    return np.random.uniform(-glorot,glorot,size=(n_inputs,n_outputs))

def softmax(scores,temp=5.0): #normalized exponential function with temperature scaling to prevent overly confident prob. for high value scores.
    ''' transforms scores to probabilites '''
    exp = np.exp(np.array(scores)/temp)
    return exp/exp.sum()

class Agent(object):
    ''' A Neural Network '''
    #Activation= Tanh
    def __init__(self, n_inputs, n_hidden, n_outputs, mutate_rate=.05, init_multiplier=1.0):
        ''' Create agent's brain '''
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.mutate_rate = mutate_rate
        self.init_multiplier = init_multiplier
        self.network = {'Layer 1' : glorot_uniform(n_inputs, n_hidden, init_multiplier), 
                        'Bias 1'  : np.zeros((1, n_hidden)),
                        'Layer 3' : glorot_uniform(n_hidden, n_hidden, init_multiplier), 
                        'Bias 3'  : np.zeros((1, n_hidden)),
                        'Layer 2' : glorot_uniform(n_hidden, n_outputs, init_multiplier),
                        'Bias 2'  : np.zeros((1, n_outputs))}
                        
    def act(self, state):
        ''' Use the network to decide on an action ''' 
        #print(state)
        if type(state) == tuple: ### MIA 
            state = state[0]  ### MIA  
        if(state.shape[0] != 1):
            state = state.reshape(1,-1)
        net = self.network
        layer_one = np.tanh(np.matmul(state,net['Layer 1']) + net['Bias 1'])
        layer_two = np.tanh(np.matmul(layer_one, net['Layer 3']) + net['Bias 3'])
        layer_three = np.tanh(np.matmul(layer_two, net['Layer 2']) + net['Bias 2'])
        return layer_three[0]
    
    def __add__(self, another):
        ''' overloads the + operator for breeding '''
        child = Agent(self.n_inputs, self.n_hidden, self.n_outputs, self.mutate_rate, self.init_multiplier)
        for key in child.network:
            n_inputs,n_outputs = child.network[key].shape
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[.5,.5])
            random = glorot_uniform(mask.shape[0],mask.shape[1]) #random weights initialized with glorot
            child.network[key] = np.where(mask==1,self.network[key],another.network[key]) #returns indices where mask =1
            mask = np.random.choice([0,1],size=child.network[key].shape,p=[1-self.mutate_rate,self.mutate_rate]) #selects 0 with 1-mutationrate prob, 1 with mut. rate prob
            child.network[key] = np.where(mask==1,child.network[key]+random,child.network[key]) # updates child networks layers with weights=child.network[key]+random when mask 1 
        return child

    
def run_trial(env,agent,verbose=False):
    ''' an agent performs 3 episodes of the env '''
    totals = []
    for _ in range(3):
        state = env.reset()
        if verbose: env.render()
        total = 0
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated
            if reward >= 100:
                env.render()
            total += reward
        totals.append(total)
    return sum(totals)/3.0

def next_generation(env,population,scores,temperature):
    ''' breeds a new generation of agents '''
    
    scores, population =  zip(*sorted(zip(scores,population),reverse=True)) #sort scores and population w.r.t. scores.
    #select the first 25% agents and mark as children 
    children = list(population[:len(population)//4])
    #fill the remaining children with the best of parents.
    #A random sample is generated from population,with probabilities returned from softmax.
    #create 2 times the size of agents remaining after 25% children are removed.
    scores = [score / min(scores) for score in scores]
    parents = list(np.random.choice(population,size=2*(len(population)-len(children)),p=softmax(scores,temperature)))
    #Breed between 2 Agent's from the above list and add it to the children list.
    children = children + [parents[i]+parents[i+1] for i in range(0,len(parents)-1,2)]
    #run the children agents and return children agents and their scores.
    scores = [run_trial(env,agent) for agent in children]

    return children,scores


def update_plot(graph, new_data):
    graph.set_xdata(np.append(graph.get_xdata(), new_data[0]))
    graph.set_ydata(np.append(graph.get_ydata(), new_data[1]))
    plt.draw()

def train(env_name, custom_dim=False):
    ''' main function '''
    # Setup environment
    env = gym.make(env_name)

    genlist=[]
    rewardlist=[]
    # network params
    n_inputs = env.observation_space.shape[0] # 24 observations
    n_actions = env.action_space.shape[0] # 4 actions
    n_hidden = 64

    if custom_dim:
        assert custom_dim > 0
        n_hidden = custom_dim

    multiplier = 5
    
    # Population params
    pop_size = 50
    mutate_rate = .1
    softmax_temp = 5.0
    
    # Training
    n_generations = 3000
    # Create agents(as per population size)
    population = [Agent(n_inputs, n_hidden, n_actions, mutate_rate, multiplier) for i in range(pop_size)]

    # Init model to save weights
    saveAgent = AgentNN(n_inputs, n_hidden, n_actions)

    if custom_dim:
        env_name = f"{env_name}_{custom_dim}"

    # Clear json statistics for training
    f = open(f'./model_rewards/{env_name}.json', 'w')
    f.close()

    best_model = population[0]

    # Run all agents in the population
    scores = [run_trial(env,agent) for agent in population]
    # Choose the best agent from the above trial and store it as best agent.
    best = [deepcopy(population[np.argmax(scores)])]
    # Create new generation and repeat for n generations
    start_time = time.time()
    bestScore = float('-inf')

    for generation in range(n_generations):
        population,scores = next_generation(env,population, scores,softmax_temp)
        best.append(deepcopy(population[np.argmax(scores)]))
        print("Generation:",generation, " Best score:",np.max(scores), "Time:", time.time() - start_time)
        if np.max(scores) > bestScore:
            best_model = population[np.argmax(scores)]

            # Save new best model
            saveAgent.load_network_and_save(best_model.network, f"model_weights/{env_name}.pt")
            bestScore = np.max(scores)
            print(f'New best model saved {bestScore}')

        # Save statistics
        f = open(f'./model_rewards/{env_name}.json', 'a')
        json.dump({"step": generation, "time": time.time() - start_time, "best_reward": np.max(scores)}, f)       
        f.write('\n')
        f.close()

        genlist += [generation]
        rewardlist += [np.max(scores)]

    
if __name__ == '__main__':
    # Train on all environments
    # BipedalWalker is on the other file
    ENVs = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4"]
    for env in ENVs:
        train(env)


    # Train InvertedDoublePendulum with different dimensions
    env = "InvertedDoublePendulum-v4"
    DIMs = [2**i for i in range(2, 7)]
    for dim in DIMs:
        train(env, dim)