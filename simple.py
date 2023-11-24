#!/usr/bin/env python
# coding: utf-8

import os
import time
import sys
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
from joblib import Parallel, delayed

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

def go(model_name, jw, n_batch=20000, batch_size=256):

    model_name = model_name + str(jw)
    if (not os.path.exists(model_name)):
        os.mkdir(model_name)

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    env = CentreOutFF(effector=effector, max_ep_duration=1.)

    policy = Policy(env.observation_space.shape[0], 128, env.n_muscles, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)

    interval   =   100

    losses = {
        'overall': [],
        'position': [],
        'angle': [],
        'lateral': [],
        'muscle': [],
        'hidden': [],
        'jerk' : []}

    for batch in tqdm(range(n_batch),
                    desc=f"Training {n_batch} batches of {batch_size}",
                    unit="batch"):

        data = run_episode(env, policy, batch_size, catch_trial_perc=50, condition='train', ff_coefficient=0.0, detach=False)
        loss, loss_values, loss_weights = cal_loss(data, env.dt)

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
            print_losses(loss_values, loss_weights, model_name, batch, weighted=True)
            data = test(model_name + "/" + model_name + "_cfg.json", model_name + "/" + model_name + "_weights")
            plot_stuff(data, model_name + "/" + model_name)

        # Update loss values in the dictionary
        losses['overall'].append(loss.item())

    save_model(env, policy, losses, model_name)
    with open(model_name + "/" + model_name + '_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print_losses(loss_values, loss_weights, model_name, batch, weighted=True)
    data = test(model_name + "/" + model_name + "_cfg.json", model_name + "/" + model_name + "_weights")
    plot_stuff(data, model_name + "/" + model_name)

if __name__ == "__main__":
    jerk_weights = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]
    model_name = "jerk_"
    n_batch = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    result = Parallel(n_jobs=len(jerk_weights))(delayed(go)(model_name=model_name, jw=jw, n_batch=n_batch, batch_size=batch_size) for jw in jerk_weights)

