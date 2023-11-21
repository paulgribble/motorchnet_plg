#!/usr/bin/env python
# coding: utf-8

import os
import time
import sys
import json
import numpy as np
import torch as th
import motornet as mn
from simple_policy import Policy
from simple_task import CentreOutFFMinJerk
from simple_utils import *
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

def go():

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())
    env = CentreOutFFMinJerk(effector=effector, max_ep_duration=1.)

    policy = Policy(env.observation_space.shape[0], 256, env.n_muscles, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=10**-3)


    batch_size =    64
    n_batch    =   100
    interval   =  1000
    model_name = "simple"

    losses = {
        'overall': [],
        'position': [],
        'angle': [],
        'lateral': [],
        'muscle': [],
        'hidden': []}

    for batch in tqdm(range(n_batch),
                    desc=f"Training {n_batch} batches of {batch_size}",
                    unit="batch"):

        data = run_episode(env, policy, batch_size, catch_trial_perc=50, condition='train', ff_coefficient=0.0, detach=False)
        loss, position_loss, muscle_loss, hidden_loss, angle_loss, lateral_loss = cal_loss(data, env.muscle.max_iso_force, env.dt, policy, test=False)

        # backward pass & update weights
        optimizer.zero_grad() 
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
        optimizer.step()

        # save weights/config/losses
        if (batch % interval == 0) and (batch != 0):
            save_model(env, policy, losses, model_name, quiet=True)
            with open('simple_data.pkl', 'wb') as f:
                pickle.dump(data, f)

        # Update loss values in the dictionary
        losses['overall'].append(loss.item())
        losses['position'].append(position_loss.item())
        losses['angle'].append(angle_loss.item())
        losses['lateral'].append(lateral_loss.item())
        losses['muscle'].append(muscle_loss.item())
        losses['hidden'].append(hidden_loss.item())

    save_model(env, policy, losses, model_name)

    return [env,policy,losses,data]

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        print(f"\npython simple.py n, where n=number of models to train\n")
    else:
        n = int(sys.argv[1])
        if (n==0):
            print("single function call")
            result = go()
        else:
            print("Parallel joblib call")
            result = Parallel(n_jobs=n)(delayed(go)() for iteration in range(n))

    print(len(result))

