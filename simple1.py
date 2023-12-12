#!/usr/bin/env python
# coding: utf-8

import os
import sys

#motornet_home = os.path.expanduser("~") + "/github/MotorNet"
#sys.path.append(motornet_home)

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

th._dynamo.config.cache_size_limit = 16 * 1024 ** 3  # 16 GB

def go(model_name, n_batch=20000, batch_size=256, interval=250, n_hidden=128):

    if (not os.path.exists(model_name)):
        os.mkdir(model_name)

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    env = CentreOutFF(effector=effector, max_ep_duration=1.)

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
        loss, losses_weighted = cal_loss(data, go_cue_time)

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
            data, _ = test(model_name + "/" + model_name + "_cfg.json", model_name + "/" + model_name + "_weights")
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
    data, _ = test(model_name + "/" + model_name + "_cfg.json", model_name + "/" + model_name + "_weights")
    plot_stuff(data, model_name + "/" + model_name, batch=batch)

if __name__ == "__main__":
    model_name = sys.argv[1]
    n_batch = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    interval = int(sys.argv[4])
    n_hidden = int(sys.argv[5])
    go(model_name=model_name, n_batch=n_batch, batch_size=batch_size, interval = interval, n_hidden=n_hidden)


