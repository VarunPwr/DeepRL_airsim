# Deep RL training for obstacle avoidance in AirSim
# Author : Varun Pawar
# E-mail : varunpwr897@gmail.com

import gym
from env import DroneEnv
import numpy as np
import math, random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib.pyplot as plt

import cv2
import csv
import statistics
import time

from exp_replay import PrioritizedExperienceReplay
from network import DQN ,CnnDQN, DuelingCnnDQN

# Argument parser
parser = argparse.ArgumentParser(description='Some input flags for the PERD3QN')
parser.add_argument('--env', default='DroneEnv', help='AirSim Quadrotor Environment by default')
parser.add_argument('--dueling', default=False, action='store_true', help='Load if required Dueling')
parser.add_argument('--load_model', default=False, action='store_true', help='Load model')
parser.add_argument('--model_path', default='log\\model\\PRDDQN_model.pth', help='Model Path')
parser.add_argument('--optimizer_path', default='log\\model\\PRDDQN_optimizer.pth', help='Optimizer Path')
parser.add_argument('--log_path', default='log\\statistics\\', help='Log Path')
parser.add_argument('--update_freq', default=10, help='Update frequency of model')
parser.add_argument('--save_freq', default=1000, help='Save frequency of model')
parser.add_argument('--num_frames', default=1000000, help='Number of Frames for training')
parser.add_argument('--batch_size', default=128, help='Training batch size')
parser.add_argument('--replay_size', default=10000, help='Replay memory size')
parser.add_argument('--replay_initial', default=1000, help='Initial untrained replay')



args = parser.parse_args()

# Log file
timestr = time.strftime("%Y%m%d-%H%M%S")
csv_name = args.log_path + 'statistics' + timestr + '.csv'
fields = ['Episodes done','Episode Reward', 'Frames Done', 'Loss', 'framerate']    
# writing to csv file  
with open(csv_name, 'w', newline = '') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

# Initialize Environment
if args.env is 'DroneEnv':
    env = DroneEnv()
else:
    env = gym.make(arg.env)

# Initialize Epsilon 
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

# Initialize Beta
beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

# Initialize NN
if args.dueling:
    current_model = DuelingCnnDQN([1,32,32], 3)
    target_model = DuelingCnnDQN([1,32,32], 3)
else:
    current_model = CnnDQN([1,32,32], 3)
    target_model  = CnnDQN([1,32,32], 3)

# USE CUDA if available
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()

# NN Parameters
maxlr = 0.0001   
optimizer = optim.Adam(current_model.parameters(), lr = maxlr)

# Training Parameters
replay_initial = args.replay_initial
replay_size = args.replay_size
replay_buffer = PrioritizedExperienceReplay(replay_size)

# Load model if continuing training
if args.load_model:
    current_model.load_state_dict(torch.load(args.model_path))
    optimizer.load_state_dict(torch.load(args.optimizer_path))

# Update target model with current model
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)

# Adjust learning rate
def adjust_learning_rate(optimizer, frame_idx):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrr = maxlr * (0.1 ** (frame_idx//replay_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrr

# Compute TD-error for a given batch
def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta) 

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))
    weights    = Variable(torch.FloatTensor(weights))

    q_values      = current_model(state.unsqueeze(1))
    next_q_values = target_model(next_state.unsqueeze(1))

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()
    
    return loss

num_frames = args.num_frames
batch_size = args.batch_size
gamma      = 0.99

no_episodes = 0
losses = [0]
all_rewards = []
episode_reward = 0
current_time = 0.001
prev_time = 0
framerate = 0
state = env.reset()
env.setObsRandom()
for frame_idx in range(1, num_frames + 1):
    framerate = (1 - math.exp(-(frame_idx-1)/1000))*framerate + math.exp(-(frame_idx-1)/1000)/(current_time - prev_time)
    prev_time = current_time
    current_time = time.time()

    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    print("-------Action:", action)
    env.setObsDynamic()

    next_state, reward, done = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    if done:
        no_episodes += 1
        state = env.reset()
        all_rewards.append(episode_reward)
        
        data = [str(no_episodes), str(episode_reward), str(frame_idx), statistics.mean(losses), framerate]
        losses = [0]
        with open(csv_name, 'a', newline = '') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
        episode_reward = 0
        env.setObsRandom()
        if no_episodes%args.update_freq is 0:
            update_target(policy_net, target_net)

        
    print('-------Frame Rate:', framerate)    
    if (len(replay_buffer) > replay_initial):
        adjust_learning_rate(optimizer, frame_idx)
        beta = beta_by_frame(frame_idx)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.item())

    if frame_idx % args.save_freq == 0:
        torch.save(current_model.state_dict(), 'PRDDQN.pth')
        torch.save(optimizer.state_dict(), 'PRDDQN_optimizer.pth')

torch.save(current_model.state_dict(), 'PRDDQN.pth')
torch.save(optimizer.state_dict(), 'PRDDQN_optimizer.pth')
print("Training ends here ...")


