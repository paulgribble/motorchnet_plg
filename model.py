import os
import sys 
from utils import create_directory, load_env
import motornet as mn
from task import CentreOutFF
from policy import Policy
import torch as th
import numpy as np
import json
from joblib import Parallel, delayed
from pathlib import Path
from utils import create_directory, plg_plots

n_hidden = 64

def train(model_num,ff_coefficient,phase,n_batch=None,condition="pretrain",directory_name=None):
  output_folder = create_directory(directory_name=directory_name)
  model_name = "model{:02d}".format(model_num)
  device = th.device("cpu")

  if phase>=1:
    # load config and weights from the previous phase
    weight_file = list(Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_weights'))[0]
    cfg_file = list(Path(output_folder).glob(f'{model_name}_phase={phase-1}_*_cfg.json'))[0]

    # load configuration
    with open(cfg_file,'r') as file:
      cfg = json.load(file)

    # environment and network
    env = load_env(CentreOutFF,cfg)
    policy = Policy(env.observation_space.shape[0], n_hidden, env.n_muscles, device=device)
    policy.load_state_dict(th.load(weight_file))

    optimizer = th.optim.SGD(policy.parameters(), lr=0.001)
    batch_size = 32
    catch_trial_perc = 0

  else:
    # environment and network
    env = load_env(CentreOutFF)    
    policy = Policy(env.observation_space.shape[0], n_hidden, env.n_muscles, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=0.001)
    batch_size = 128
    catch_trial_perc = 50
  
  # Define Loss function
  def l1(x, y):
    """L1 loss"""
    return th.mean(th.sum(th.abs(x - y), dim=-1))

  # Train network
  losses = []
  position_loss = []
  interval = 100

  for batch in range(n_batch):

    # check if you want to load a model TODO
    h = policy.init_hidden(batch_size=batch_size)

    obs, info = env.reset(condition        = condition,
                          catch_trial_perc = catch_trial_perc,
                          ff_coefficient   = ff_coefficient,
                          options          = {'batch_size':batch_size})
    terminated = False

    # initial positions and targets
    xy = [info['states']['cartesian'][:, None, :]]
    tg = [info["goal"][:, None, :]]
    all_actions = []
    all_muscle = []
    all_hidden = []

    # simulate whole episode
    while not terminated:  # will run until `max_ep_duration` is reached
      action, h = policy(obs,h)
      obs, reward, terminated, truncated, info = env.step(action=action)

      xy.append(info['states']['cartesian'][:, None, :])  # trajectories
      tg.append(info["goal"][:, None, :])  # targets
      all_actions.append(action[:, None, :])
      all_muscle.append(info['states']['muscle'][:,0,None,:])
      all_hidden.append(h[0,:,None,:])

    # concatenate into a (batch_size, n_timesteps, xy) tensor
    xy = th.cat(xy, axis=1)
    tg = th.cat(tg, axis=1)
    all_hidden = th.cat(all_hidden, axis=1)
    all_actions = th.cat(all_actions, axis=1)
    all_muscle = th.cat(all_muscle, axis=1)

    # calculate losses
    cartesian_loss = l1(xy[:,:,0:2], tg)
    action_loss = 1e-5 * th.sum(th.abs(all_actions))
    hidden_loss = 1e-6 * th.sum(th.abs(all_hidden))
    hidden_diff_loss = 1e-8 * th.sum(th.abs(th.diff(all_hidden, dim=1)))
    #muscle_loss = 0.1 * th.mean(th.sum(th.square(all_muscle), dim=-1))
    recurrent_loss = 1e-4 * th.sum(th.square(policy.gru.weight_hh_l0))

    loss = cartesian_loss + action_loss + hidden_loss + hidden_diff_loss + recurrent_loss
    
    # backward pass & update weights
    optimizer.zero_grad() 
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)  # important!
    optimizer.step()
    losses.append(loss.item())
    position_loss.append(cartesian_loss.item())

    if (batch % interval == 0) and (batch != 0):
      print("Phase {}/{}, Batch {}/{} Done, mean policy loss: {}".format(phase,4,batch, n_batch, sum(losses[-interval:])/interval))

  # Save model
  weight_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
  log_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
  cfg_file = os.path.join(output_folder, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")


  # save model weights
  th.save(policy.state_dict(), weight_file)

  # save training history (log)
  with open(log_file, 'w') as file:
    #json.dump(losses, file)
    json.dump({'losses':losses,'position_loss':position_loss}, file)

  # save environment configuration dictionary
  cfg = env.get_save_config()
  with open(cfg_file, 'w') as file:
    json.dump(cfg, file)

  print("done.")

def test(cfg_file, weight_file, ff_coefficient=None):
  device = th.device("cpu")

  # load configuration
  with open(cfg_file,'r') as file:
    cfg = json.load(file)

  if ff_coefficient is None:
    ff_coefficient=cfg['ff_coefficient']
    
  # environment and network
  env = load_env(CentreOutFF, cfg)
  policy = Policy(env.observation_space.shape[0], n_hidden, env.n_muscles, device=device)
  policy.load_state_dict(th.load(weight_file))

  batch_size = 8
  # initialize batch
  obs, info = env.reset(condition        = "test",
                        catch_trial_perc = 0,
                        options          = {'batch_size':batch_size},
                        ff_coefficient   = ff_coefficient)

  h = policy.init_hidden(batch_size = batch_size)
  terminated = False

  # initial positions and targets
  xy = [info["states"]["fingertip"][:, None, :]]
  tg = [info["goal"][:, None, :]]

  # simulate whole episode
  while not terminated:  # will run until `max_ep_duration` is reached
    action, h = policy(obs, h)

    obs, reward, terminated, truncated, info = env.step(action=action)  
    xy.append(info["states"]["fingertip"][:,None,:])  # trajectories
    tg.append(info["goal"][:,None,:])  # targets

  # concatenate into a (batch_size, n_timesteps, xy) tensor
  xy = th.detach(th.cat(xy, axis=1))
  tg = th.detach(th.cat(tg, axis=1))

  return xy, tg


if __name__ == "__main__":
    ## training single network - use for debugging
    # model_num = int(sys.argv[1])
    # ff_coefficient = float(sys.argv[2])
    # phase = int(sys.argv[3])
    # directory_name = sys.argv[4]
    # train(model_num,ff_coefficient,phase,directory_name)

    trainall = int(sys.argv[1])

    if trainall:
      directory_name = sys.argv[2]

      n_jobs = 10
      iter_list = range(n_jobs)
      while len(iter_list) > 0:
          these_iters = iter_list[0:n_jobs]
          iter_list = iter_list[n_jobs:]
          # pretraining the network using ADAM
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,
                                                                    0,
                                                                    0,
                                                                    n_batch=40000,
                                                                    condition='pretrain',
                                                                    directory_name=directory_name) 
                                                     for iteration in these_iters)
          # NF1
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,
                                                                    0,
                                                                    1,
                                                                    n_batch=100,
                                                                    condition='pretrain',
                                                                    directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF1
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,
                                                                    10,
                                                                    2,
                                                                    n_batch=1500,
                                                                    condition='pretrain',
                                                                    directory_name=directory_name) 
                                                     for iteration in these_iters)
          # NF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,
                                                                    0,
                                                                    3,
                                                                    n_batch=500,
                                                                    condition='pretrain',
                                                                    directory_name=directory_name) 
                                                     for iteration in these_iters)
          # FF2
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,
                                                                    10,
                                                                    4,
                                                                    n_batch=1500,
                                                                    condition='pretrain',
                                                                    directory_name=directory_name) 
                                                     for iteration in these_iters)
          
    else: ## training networks for each phase separately
      ff_coefficient = float(sys.argv[2])
      phase = int(sys.argv[3])
      n_batch = int(sys.argv[4])
      condition = sys.argv[5]
      directory_name = sys.argv[6]
      n_jobs = int(sys.argv[7])

      iter_list = range(n_jobs)
      while len(iter_list) > 0:
          these_iters = iter_list[0:n_jobs]
          iter_list = iter_list[n_jobs:]
          result = Parallel(n_jobs=len(these_iters))(delayed(train)(iteration,ff_coefficient,phase,n_batch=n_batch,condition=condition,directory_name=directory_name) 
                                                     for iteration in these_iters)

    data_dir = create_directory(directory_name=directory_name)
    model_num = 0
    model_name = "model{:02d}".format(model_num)
    phase: int = 0
    ff_coefficient = int(str(0))
    weight_file = os.path.join(data_dir, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_weights")
    log_file = os.path.join(data_dir, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_log.json")
    cfg_file = os.path.join(data_dir, f"{model_name}_phase={phase}_FFCoef={ff_coefficient}_cfg.json")
    xy, tg = test(cfg_file,weight_file,ff_coefficient=0)
    fig,ax = plg_plots(data_dir, num_model=n_jobs, w=10, figsize=(8,8), init_phase=1, xy=xy, target_xy=tg)
    fig.savefig(os.path.join(data_dir, 'plots.png'), dpi=150)

