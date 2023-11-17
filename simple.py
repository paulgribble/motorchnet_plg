import os
import time
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
from task import CentreOutFFMinJerk
from policy import Policy
from tqdm import tqdm

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

def plot_simulations(xy, target_xy, figsize=(5,3)):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time
    plotor(axis=ax, cart_results=xy)

    ax.scatter(target_x, target_y)
    
    fig.tight_layout()
    return fig, ax

def plot_activation(all_hidden, all_muscles):
    fig, ax = plt.subplots(nrows=8,ncols=2,figsize=(6,10))

    x = np.linspace(0, 1, 100)

    for i in range(8):
        ax[i,0].plot(x,np.array(all_muscles[i,:,:]))
        ax[i,1].plot(x,np.array(all_hidden[i,:,:]))
        
        ax[i,0].set_ylabel('muscle act (au)')
        ax[i,1].set_ylabel('hidden act (au)')
        ax[i,0].set_xlabel('time (s)')
        ax[i,1].set_xlabel('time (s)')
    fig.tight_layout()
    return fig, ax


def run_episode(env, policy, batch_size=1, catch_trial_perc=50, condition='train', ff_coefficient=None, detach=False):

  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size})
  terminated = False

  # Initialize a dictionary to store lists
  data = {
      'xy': [],
      'tg': [],
      'vel': [],
      'all_actions': [],
      'all_hidden': [],
      'all_muscle': [],
      'all_force': [],
  }

  while not terminated:
      # Append data to respective lists
      data['all_hidden'].append(h[0, :, None, :])
      data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])

      action, h = policy(obs, h)
      obs, _, terminated, _, info = env.step(action=action)

      data['xy'].append(info["states"]["fingertip"][:, None, :])
      data['tg'].append(info["goal"][:, None, :])
      data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
      data['all_actions'].append(action[:, None, :])
      data['all_force'].append(info['states']['muscle'][:, 6, None, :])

  # Concatenate the lists
  for key in data:
      data[key] = th.cat(data[key], axis=1)

  if detach:
      # Detach tensors if needed
      for key in data:
          data[key] = th.detach(data[key])

  return data


def test(cfg_file, weight_file, ff_coefficient=None):

    device = th.device("cpu")

    # load configuration
    cfg = json.load(open(cfg_file, 'r'))

    if ff_coefficient is None:
        ff_coefficient=cfg['ff_coefficient']

    # environment
    name = cfg['name']
    # effector
    muscle_name = cfg['effector']['muscle']['name']
    timestep = cfg['effector']['dt']
    muscle = getattr(mn.muscle,muscle_name)()
    effector = mn.effector.RigidTendonArm26(muscle=muscle,timestep=timestep) 
    # delay
    proprioception_delay = cfg['proprioception_delay']*cfg['dt']
    vision_delay = cfg['vision_delay']*cfg['dt']
    # noise
    action_noise = cfg['action_noise'][0]
    proprioception_noise = cfg['proprioception_noise'][0]
    vision_noise = cfg['vision_noise'][0]
    # initialize environment
    max_ep_duration = cfg['max_ep_duration']
    env = CentreOutFFMinJerk(effector=effector,max_ep_duration=max_ep_duration,name=name,
               action_noise=action_noise,proprioception_noise=proprioception_noise,
               vision_noise=vision_noise,proprioception_delay=proprioception_delay,
               vision_delay=vision_delay)

    # network
    w = th.load(weight_file)
    num_hidden = int(w['gru.weight_ih_l0'].shape[0]/3)
    if 'h0' in w.keys():
        policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=True)
    else:
        policy = Policy(env.observation_space.shape[0], num_hidden, env.n_muscles, device=device, learn_h0=False)
    policy.load_state_dict(w)

    # Run episode
    data = run_episode(env,policy,8,0,'test',ff_coefficient=ff_coefficient,detach=True)
    
    return data

cfg_file = "simple_cfg.json"
weight_file = "simple_weights"

# TEST NETWORK ON CENTRE-OUT

data = test(cfg_file, weight_file)

fig, ax  = plot_simulations(xy=data['xy'],target_xy=data['tg'], figsize=(8,6))

fig, ax = plot_activation(data['all_hidden'], data['all_muscle'])
