import os
import datetime
import matplotlib.pyplot as plt
import motornet as mn
import numpy as np
from pathlib import Path
import json


def create_directory(directory_name=None):
    if directory_name is None:
        directory_name = datetime.datetime.now().date().isoformat()

    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Create the full directory path
    directory_path = os.path.join(home_directory, "Documents", "Data", directory_name)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    # Return the created directory's name (whether it was newly created or already existed)
    return directory_path

# average over non-overlapping windows size w;
# throws away tail of x that doesn't divide evenly into w
def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def plot_training_log(log):
    ax = plt.subplot(1,1,1)

    ax.semilogy(log)

    ax.set_ylabel("Loss")
    ax.set_xlabel("Batch #")
    return ax

def plg_plots(data_dir,num_model,w,figsize,init_phase=1, xy, target_xy):
    position_loss_NF1 = []
    position_loss_FF1 = []
    position_loss_NF2 = []
    position_loss_FF2 = []

    # Loop through each model
    for m in range(num_model):

        model_name = "model{:02d}".format(m)


        log_file1 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase}_*_log.json'))[0]
        log_file2 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+1}_*_log.json'))[0]
        log_file3 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+2}_*_log.json'))[0]
        log_file4 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+3}_*_log.json'))[0]
        
        position_loss_NF1_ = json.load(open(log_file1,'r'))
        position_loss_FF1_ = json.load(open(log_file2,'r'))
        position_loss_NF2_ = json.load(open(log_file3,'r'))
        position_loss_FF3_ = json.load(open(log_file4,'r'))
        
        # Append data for each model
        position_loss_NF1.append(position_loss_NF1_['position_loss'])
        position_loss_FF1.append(position_loss_FF1_['position_loss'])
        position_loss_NF2.append(position_loss_NF2_['position_loss'])
        position_loss_FF2.append(position_loss_FF3_['position_loss'])


    # Calculate window averages for all models
    NF1w = [window_average(np.array(loss), w) for loss in position_loss_NF1]
    FF1w = [window_average(np.array(loss), w) for loss in position_loss_FF1]
    NF2w = [window_average(np.array(loss), w) for loss in position_loss_NF2]
    FF2w = [window_average(np.array(loss), w) for loss in position_loss_FF2]


    # Calculate the mean and standard deviation across models
    NF1_mean = np.mean(NF1w, axis=0)
    FF1_mean = np.mean(FF1w, axis=0)
    NF2_mean = np.mean(NF2w, axis=0)
    FF2_mean = np.mean(FF2w, axis=0)

    NF1_se = np.std(NF1w, axis=0) / np.sqrt(num_model)
    FF1_se = np.std(FF1w, axis=0) / np.sqrt(num_model)
    NF2_se = np.std(NF2w, axis=0) / np.sqrt(num_model)
    FF2_se = np.std(FF2w, axis=0) / np.sqrt(num_model)

    x1w = np.arange(1,np.shape(NF1w)[1]+1)
    x2w = np.arange(1,np.shape(FF1w)[1]+1) + x1w[-1]
    x3w = np.arange(1,np.shape(NF2w)[1]+1) + x2w[-1]
    x4w = np.arange(1,np.shape(FF2w)[1]+1) + x3w[-1]


    fig,ax = plt.subplot_mosaic([['upper left', 'upper right'],
                               ['lower', 'lower']],
                               figsize=figsize, layout="constrained")
    ax[0,0].plot(x1w,NF1_mean,'k.-',label='NF1')
    ax[0,0].fill_between(x1w, NF1_mean - NF1_se, NF1_mean + NF1_se, color='gray', alpha=0.5)
    ax[0,0].plot(x2w,FF1_mean,'g.-',label='FF1')
    ax[0,0].fill_between(x2w, FF1_mean - FF1_se, FF1_mean + FF1_se, color='green', alpha=0.5)
    ax[0,0].plot(x3w,NF2_mean,'k.-',label='NF2')
    ax[0,0].fill_between(x3w, NF2_mean - NF2_se, NF2_mean + NF2_se, color='gray', alpha=0.5)
    ax[0,0].plot(x4w,FF2_mean,'r.-',label='FF2')
    ax[0,0].fill_between(x4w, FF2_mean - FF2_se, FF2_mean + FF2_se, color='red', alpha=0.5)
    ax[0,0].set_title(f"window avg size = {w}")
    ax[0,0].legend()

    ax[0,1].plot(FF1_mean,'g.-',label='FF1')
    ax[0,1].plot(FF2_mean,'r.-',label='FF2')
    ax[0,1].legend()

    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    ax[1,(0,1)].set_ylim([0.3, 0.65])
    ax[1,(0,1)].set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time

    plotor(axis=ax[0,1], cart_results=xy)
    ax[0,1].scatter(target_x, target_y)

    return fig, ax


def plot_simulations(xy, target_xy):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fig,ax = plt.subplots(1,1,figsize=(5,3))

    ax.set_ylim([0.3, 0.65])
    ax.set_xlim([-0.3, 0.])

    plotor = mn.plotor.plot_pos_over_time

    plotor(axis=ax, cart_results=xy)
    ax.scatter(target_x, target_y)

    #plt.subplot(1,2,2)
    #plt.ylim([-0.1, 0.1])
    #plt.xlim([-0.1, 0.1])
    #plotor(axis=plt.gca(), cart_results=xy - target_xy)
    #plt.axhline(0, c="grey")
    #plt.axvline(0, c="grey")
    #plt.xlabel("X distance to target")
    #plt.ylabel("Y distance to target")
    return fig, ax

def load_env(task,cfg=None):

    if cfg is None:

        name = 'env'

        action_noise         = 1e-4
        proprioception_noise = 1e-3
        vision_noise         = 1e-4
        vision_delay         = 0.05
        proprioception_delay = 0.02

        # Define task and the effector
        effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscleThelen())

        max_ep_duration = 1.0
    else:
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


    env = task(effector=effector,max_ep_duration=max_ep_duration,name=name,
               action_noise=action_noise,proprioception_noise=proprioception_noise,
               vision_noise=vision_noise,proprioception_delay=proprioception_delay,
               vision_delay=vision_delay)

    return env


def plot_learning(data_dir,num_model=10,w=100,figsize=(6,10),init_phase=1):
    position_loss_NF1 = []
    position_loss_FF1 = []
    position_loss_NF2 = []
    position_loss_FF2 = []

    # Loop through each model
    for m in range(num_model):

        model_name = "model{:02d}".format(m)


        log_file1 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase}_*_log.json'))[0]
        log_file2 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+1}_*_log.json'))[0]
        log_file3 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+2}_*_log.json'))[0]
        log_file4 = list(Path(data_dir).glob(f'{model_name}_phase={init_phase+3}_*_log.json'))[0]
        
        position_loss_NF1_ = json.load(open(log_file1,'r'))
        position_loss_FF1_ = json.load(open(log_file2,'r'))
        position_loss_NF2_ = json.load(open(log_file3,'r'))
        position_loss_FF3_ = json.load(open(log_file4,'r'))
        
        # Append data for each model
        position_loss_NF1.append(position_loss_NF1_['position_loss'])
        position_loss_FF1.append(position_loss_FF1_['position_loss'])
        position_loss_NF2.append(position_loss_NF2_['position_loss'])
        position_loss_FF2.append(position_loss_FF3_['position_loss'])


    # Calculate window averages for all models
    NF1w = [window_average(np.array(loss), w) for loss in position_loss_NF1]
    FF1w = [window_average(np.array(loss), w) for loss in position_loss_FF1]
    NF2w = [window_average(np.array(loss), w) for loss in position_loss_NF2]
    FF2w = [window_average(np.array(loss), w) for loss in position_loss_FF2]


    # Calculate the mean and standard deviation across models
    NF1_mean = np.mean(NF1w, axis=0)
    FF1_mean = np.mean(FF1w, axis=0)
    NF2_mean = np.mean(NF2w, axis=0)
    FF2_mean = np.mean(FF2w, axis=0)

    NF1_se = np.std(NF1w, axis=0) / np.sqrt(num_model)
    FF1_se = np.std(FF1w, axis=0) / np.sqrt(num_model)
    NF2_se = np.std(NF2w, axis=0) / np.sqrt(num_model)
    FF2_se = np.std(FF2w, axis=0) / np.sqrt(num_model)

    x1w = np.arange(1,np.shape(NF1w)[1]+1)
    x2w = np.arange(1,np.shape(FF1w)[1]+1) + x1w[-1]
    x3w = np.arange(1,np.shape(NF2w)[1]+1) + x2w[-1]
    x4w = np.arange(1,np.shape(FF2w)[1]+1) + x3w[-1]


    fig,ax = plt.subplots(2,1,figsize=figsize)
    ax[0].plot(x1w,NF1_mean,'k.-',label='NF1')
    ax[0].fill_between(x1w, NF1_mean - NF1_se, NF1_mean + NF1_se, color='gray', alpha=0.5)
    ax[0].plot(x2w,FF1_mean,'g.-',label='FF1')
    ax[0].fill_between(x2w, FF1_mean - FF1_se, FF1_mean + FF1_se, color='green', alpha=0.5)
    ax[0].plot(x3w,NF2_mean,'k.-',label='NF2')
    ax[0].fill_between(x3w, NF2_mean - NF2_se, NF2_mean + NF2_se, color='gray', alpha=0.5)
    ax[0].plot(x4w,FF2_mean,'r.-',label='FF2')
    ax[0].fill_between(x4w, FF2_mean - FF2_se, FF2_mean + FF2_se, color='red', alpha=0.5)
    ax[0].set_title(f"window avg size = {w}")
    ax[0].legend()


    ax[1].plot(FF1_mean,'g.-',label='FF1')
    ax[1].plot(FF2_mean,'r.-',label='FF2')
    ax[1].legend()


    return fig, ax

def plot_prelearning(data_dir,num_model=10,phase=0,w=1000,figsize=(6,10)):
    position_loss_NF1 = []

    # Loop through each model
    for m in range(num_model):

        model_name = "model{:02d}".format(m)
        log_file1 = list(Path(data_dir).glob(f'{model_name}_phase={phase}_*_log.json'))[0]

        position_loss_NF1_ = json.load(open(log_file1,'r'))
        
        # Append data for each model
        position_loss_NF1.append(position_loss_NF1_['position_loss'])

    # Calculate window averages for all models
    NF1w = [window_average(np.array(loss), w) for loss in position_loss_NF1]

    # Calculate the mean and standard deviation across models
    NF1_mean = np.mean(NF1w, axis=0)
    NF1_std = np.std(NF1w, axis=0)


    x1w = np.arange(1,np.shape(NF1w)[1]+1)


    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(x1w,NF1_mean,'k.-',label='NF1')
    ax.fill_between(x1w, NF1_mean - NF1_std, NF1_mean + NF1_std, color='gray', alpha=0.5)
    ax.legend()

    return fig, ax 