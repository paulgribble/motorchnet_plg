from motornet.policy import ModularPolicyGRU

# PFC, PMd, M1, S1, PPC, Spinal
proportion_excitatory = None
vision_mask = [0, 0, 0, 0, 1, 0]
proprio_mask = [0, 0, 0, 0, 0, 1]
task_mask = [0, 0, 0, 0, 1, 0]
c1 = 0.2
c2 = 0.05
c3 = 0.01
c0 = 0.5
connectivity_mask = np.array([[c0, c1, c3, 0, c1, 0],
                              [c1, c0, c1, c3, c1, 0],
                              [c2, c1, c0, c1, c3, c3],
                              [0, 0, c3, c0, c3, c1],
                              [c1, c1, c3, c1, c0, 0],
                              [0, 0, c1, 0, 0, c1]])
connectivity_delay = np.zeros_like(connectivity_mask)
output_mask = [0, 0, 0, 0, 0, 1]
module_sizes = [64, 64, 64, 64, 64, 16]
spectral_scaling = 1.
# input sparsity
vision_dim = np.arange(env.get_vision().shape[1])
proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1
task_dim = np.arange(inputs['inputs'].shape[2]) + proprio_dim[-1] + 1
policy = ModularPolicyGRU(env.observation_space.shape[0] + inputs['inputs'].shape[2], module_sizes, env.n_muscles, 
                vision_dim=vision_dim, proprio_dim=proprio_dim, task_dim=task_dim, 
                vision_mask=vision_mask, proprio_mask=proprio_mask, task_mask=task_mask,
                connectivity_mask=connectivity_mask, output_mask=output_mask, connectivity_delay=connectivity_delay,
                proportion_excitatory=proportion_excitatory, input_gain=1,
                spectral_scaling=spectral_scaling, device=device, activation='tanh')
optimizer = th.optim.Adam(policy.parameters(), lr=3e-3)
scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.9999)


