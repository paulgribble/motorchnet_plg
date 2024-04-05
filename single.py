#!/usr/bin/env python
# coding: utf-8

import os
import sys

import time
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
from simple_policy import Policy
from simple_task import CentreOutFF
from simple_utils import *
from tqdm import tqdm
import pickle

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

def go(model_name, n_batch=20000, batch_size=256, interval=250, n_hidden=128, loss_weights=None):

    if (not os.path.exists(model_name)):
        os.mkdir(model_name)

    with open(model_name + "/" + model_name + "___launchcmd.txt", "w") as f:
        print(f"{sys.argv}", file=f)
        print(f"model_name : {model_name}\n"
              f"n_batch    :{n_batch:6d}\n"
              f"batch_size :{batch_size:6d}\n"
              f"interval   :{interval:6d}\n"
              f"n_hidden   :{n_hidden:6d}\n"
              f"loss_weights :{loss_weights}\n"
              , file=f)

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    env = CentreOutFF(effector=effector, max_ep_duration=1.5)

    policy = Policy(env.observation_space.shape[0], n_hidden, env.n_muscles, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

    losses = {
        'overall': [],
        'position': [],
        'muscle': [],
        'muscle_derivative': [],
        'hidden': [],
        'hidden_derivative': [],
        'jerk' : []}

    for batch in tqdm(range(n_batch),
                    desc=f"Training {n_batch} batches of {batch_size}",
                    unit="batch"):

        data, go_cue_time = run_episode(env, policy, batch_size, catch_trial_perc=50, condition='train', ff_coefficient=0.0, detach=False)
        loss, losses_weighted = cal_loss(data=data, go_cue_time=go_cue_time, dt=env.dt, loss_weights=loss_weights)

        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()

        # save weights/config/losses
        if (batch % interval == 0) and (batch != 0):
            save_model(env, policy, losses, model_name, quiet=True)
            with open(model_name + "/" + model_name + '_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            print_losses(losses_weighted=losses_weighted, model_name=model_name, batch=batch)
            data, _ = test(cfg_file  = model_name + "/" + model_name + "_cfg.json",
                        weight_file  = model_name + "/" + model_name + "_weights",
                        loss_weights = loss_weights)
            plot_stuff(data, model_name + "/" + model_name, batch=batch)

        # Update loss values in the dictionary
        losses['overall'].append(loss.item())
        losses['position'].append(losses_weighted['position'].item())
        losses['muscle'].append(losses_weighted['muscle'].item())
        losses['muscle_derivative'].append(losses_weighted['muscle_derivative'].item())
        losses['hidden'].append(losses_weighted['hidden'].item())
        losses['hidden_derivative'].append(losses_weighted['hidden_derivative'].item())
        losses['jerk'].append(losses_weighted['jerk_loss'].item())

    save_model(env, policy, losses, model_name)
    with open(model_name + "/" + model_name + '_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print_losses(losses_weighted=losses_weighted, model_name=model_name, batch=batch)
    data, _ = test(cfg_file     = model_name + "/" + model_name + "_cfg.json",
                   weight_file  = model_name + "/" + model_name + "_weights",
                   loss_weights = loss_weights)
    plot_stuff(data, model_name + "/" + model_name, batch=batch)


if __name__ == "__main__":

    model_name = sys.argv[1]
    n_batch = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    interval = int(sys.argv[4])
    n_hidden = int(sys.argv[5])

    loss_weights = np.array([1e+3,   # position
                             5e-2,   # muscle
                             1e-8,   # muscle_derivative
                             1e-3,   # hidden # was 1e-4
                             1e-8,   # hidden_derivative
                             1e-7])  # jerk on hand path

    go(model_name   = model_name, 
       n_batch      = n_batch, 
       batch_size   = batch_size, 
       interval     = interval, 
       n_hidden     = n_hidden,
       loss_weights = loss_weights)

