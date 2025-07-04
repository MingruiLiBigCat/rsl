# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from modules.vae import VAE

class ActorCriticHDS(nn.Module):
    """
    Actor-Critic model with Hierarchical Discrete Structure (HDS) for robotic control.
    
    Key Components:
    1. Discrete Hybrid Automata (DHA): Classifies modes from observations
    2. Transition Dynamics (TsDyn): Set of VAEs for mode-specific temporal representations
    3. Policy Network (Actor): Generates Beta-distributed actions
    4. Value Networks (Critics): Multiple value estimators
    
    Tensor Shapes:
    - Input Observations: (batch_size, num_actor_obs)
    - DHA Output: 
        mode_latent: (batch_size, num_modes) one-hot encoded
        prob: (batch_size, num_modes) probability distribution
    - TsDyn Output (representation): (batch_size, tsdyn_latent_dims)
    - Actor Input: (batch_size, tsdyn_latent_dims + num_proprio)
    - Actor Output: (batch_size, num_actions*2)  # alpha and beta parameters
    - Critic Input: (batch_size, num_critic_obs)
    - Critic Output: (batch_size, 1)
    
    Args:
        num_actor_obs (int): Dimension of actor observations
        num_critic_obs (int): Dimension of critic observations
        num_actions (int): Dimension of action space
        num_proprio (int): Dimension of proprioceptive features
        num_recon (int): Dimension of reconstructed features in VAEs
        history_len (int): Temporal window length for VAEs
        num_modes (int): Number of discrete modes for DHA
        actor_hidden_dims (list[int]): Actor MLP layer dimensions
        critic_hidden_dims (list[int]): Critic MLP layer dimensions
        dha_hidden_dims (list[int]): DHA MLP layer dimensions
        tsdyn_hidden_dims (list[int]): VAE encoder/decoder dimensions
        tsdyn_latent_dims (int): Latent dimension for VAEs
        cfg (object): Configuration object
    """
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_proprio,
                        num_recon,
                        history_len, 
                        num_modes = 3,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        dha_hidden_dims = [256, 64, 32],
                        tsdyn_hidden_dims = [256, 128, 64],
                        tsdyn_latent_dims = 32,
                        cfg = None,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticHDS, self).__init__()
        self.cfg = cfg
        self.num_proprio = num_proprio
        self.num_recon = num_recon
        self.history_len = history_len
        self.clip_range = 0.35

        num_his_obs = num_actor_obs
        mlp_input_dim_a = tsdyn_latent_dims + self.num_proprio
        mlp_input_dim_c = num_critic_obs
        # beta action(alpha and beta)
        num_actions *= 2

        # Actor Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(nn.ELU())
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                actor_layers.append(SoftplusWithOffset())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(nn.ELU())
        print(actor_layers)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(nn.ELU())
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(nn.ELU())
        self.critic = nn.Sequential(*critic_layers)

        # glide critic
        glide_critic_layers = []
        glide_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        glide_critic_layers.append(nn.ELU())
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                glide_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                glide_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                glide_critic_layers.append(nn.ELU())
        self.glide_critic = nn.Sequential(*glide_critic_layers)

        # push critic
        push_critic_layers = []
        push_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        push_critic_layers.append(nn.ELU())
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                push_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                push_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                push_critic_layers.append(nn.ELU())
        self.push_critic = nn.Sequential(*push_critic_layers)

        # regulation critic
        reg_critic_layers = []
        reg_critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        reg_critic_layers.append(nn.ELU())
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                reg_critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                reg_critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                reg_critic_layers.append(nn.ELU())
        self.reg_critic = nn.Sequential(*reg_critic_layers)

        # Discrete Hybrid automata
        DHA_layers = []
        DHA_input_dim = num_actor_obs
        DHA_out_put_dim = num_modes
        DHA_layers.append(nn.Linear(DHA_input_dim, dha_hidden_dims[0]))
        DHA_layers.append(nn.ELU())
        for l in range(len(dha_hidden_dims)):
            if l == len(dha_hidden_dims) - 1:
                DHA_layers.append(nn.Linear(dha_hidden_dims[l], DHA_out_put_dim))
                DHA_layers.append(StraightThroughSoftmax())
            else:
                DHA_layers.append(nn.Linear(dha_hidden_dims[l], dha_hidden_dims[l + 1]))
                DHA_layers.append(nn.ELU())
        self.DHA = nn.Sequential(*DHA_layers)

        # Transition dynamics net
        self.TsDyn_modules = nn.ModuleList()
        TsDyn_input_dim = num_his_obs
        TsDyn_output_dim = self.num_recon
        for i in range (num_modes):
            self.TsDyn_modules.append(VAE(TsDyn_input_dim, TsDyn_output_dim, tsdyn_hidden_dims, tsdyn_latent_dims, self.history_len, kl_w=0.9, prior_mu=0))


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Glide Critic MLP: {self.glide_critic}")
        print(f"Push Critic MLP: {self.push_critic}")
        print(f"Reg Critic MLP: {self.reg_critic}")

        # Action noise
        self.distribution = None
        # disable args validation for speedup
        Beta.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        """Resets internal states (not used in this model)."""
        pass

    def forward(self):
        """Not implemented (use specific methods instead)."""
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """
        Mean of current action distribution.
        
        Returns:
            tensor: (batch_size, num_actions)
        """
        return self.distribution.mean

    @property
    def action_std(self):
        """
        Standard deviation of current action distribution.
        
        Returns:
            tensor: (batch_size, num_actions)
        """
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """
        Entropy of current action distribution.
        
        Returns:
            tensor: (batch_size,)
        """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """
        Updates the action distribution based on current observations.
        
        Processing Flow:
        1. DHA: observations (batch_size, num_actor_obs) → mode_latent (batch_size, num_modes)
        2. TsDyn: Each VAE processes observations → representations (batch_size, tsdyn_latent_dims)
        3. Weighted combination: (batch_size, 1, num_modes) @ (batch_size, num_modes, tsdyn_latent_dims)
                                → (batch_size, tsdyn_latent_dims)
        4. Actor: [proprio (batch_size, num_proprio), representation] → 
                 action parameters (batch_size, num_actions*2)
        5. Creates Beta distribution from parameters
        
        Args:
            observations (tensor): Input tensor of shape (batch_size, num_actor_obs)
        """
        assert not torch.isnan(observations).any(), "Observations contain NaN values!"
        # get one hot latent
        mode_latent, prob = self.DHA(observations)
        mode_latent = mode_latent.detach()
        representation_list = []
        # get transition dynamics representation
        for i, sub_net in enumerate(self.TsDyn_modules):
            representation_list.append(sub_net.get_representation(observations).unsqueeze(1))
        representation = torch.cat(representation_list, dim=1)
        representation = torch.bmm(mode_latent.unsqueeze(1), representation).squeeze(1)
        # get action from obs and td representation
        input = torch.cat((observations[:, -self.num_proprio:], representation), dim=-1)
        action = self.actor(input)
        assert not torch.isnan(action).any(), "Action contain NaN values!"
        alpha = torch.clamp(action[:, 0::2], min=1e-6)
        beta = torch.clamp(action[:, 1::2], min=1e-6)
        self.distribution = Beta(alpha, beta)

    def act(self, observations, **kwargs):
        """
        Samples actions from current distribution.
        
        Args:
            observations (tensor): Shape (batch_size, num_actor_obs)
        
        Returns:
            tensor: Actions in range [-clip_range, clip_range], shape (batch_size, num_actions)
        """
        self.update_distribution(observations)
        return self.distribution.sample() * self.clip_range * 2 - self.clip_range
    
    def get_actions_log_prob(self, actions):
        """
        Computes log probability of given actions under current distribution.
        
        Args:
            actions (tensor): Shape (batch_size, num_actions) in range [-clip_range, clip_range]
        
        Returns:
            tensor: Log probabilities of shape (batch_size,)
        """
        actions = torch.clamp(actions, min=-3+1e-6, max=3-1e-6)
        original_actions = (actions + self.clip_range) / (self.clip_range * 2)
        log_prob = self.distribution.log_prob(original_actions).sum(dim=-1)
        log_prob -= torch.log(torch.tensor(self.clip_range * 2))
        return log_prob

    def act_inference(self, observations):
        """
        Action prediction without sampling (deterministic mode).
        
        Processing Flow:
        1. DHA: observations → mode_latent (batch_size, num_modes)
        2. TsDyn: representations (batch_size, num_modes, tsdyn_latent_dims)
        3. Weighted combination → (batch_size, tsdyn_latent_dims)
        4. Actor: mean action (batch_size, num_actions*2)
        5. Transforms to range [-3, 3] via alpha/(alpha+beta)*6 - 3
        
        Args:
            observations (tensor): Shape (batch_size, num_actor_obs)
        
        Returns:
            tuple: 
                actions: (batch_size, num_actions) in range [-3, 3]
                mode_latent: (batch_size, num_modes)
        """
        mode_latent, prob = self.DHA(observations)
        representation_list = []
        for i, sub_net in enumerate(self.TsDyn_modules):
            representation_list.append(sub_net.get_representation(observations).unsqueeze(1))
        representation = torch.cat(representation_list, dim=1)
        representation = torch.bmm(mode_latent.unsqueeze(1), representation).squeeze(1)
        input = torch.cat((observations[:, -self.num_proprio:], representation), dim=-1)
        actions_mean = self.actor(input)
        alpha = torch.clamp(actions_mean[:, 0::2], min=1e-6)
        beta = torch.clamp(actions_mean[:, 1::2], min=1e-6)
        return (alpha/(alpha + beta))*6 - 3, mode_latent

    def evaluate(self, critic_observations, **kwargs):
        """
        Evaluates state value using multiple critics.
        
        Args:
            critic_observations (tensor): Shape (batch_size, num_critic_obs)
        
        Returns:
            tuple: Four value tensors each of shape (batch_size, 1)
        """
        value = self.critic(critic_observations)
        glide_value = self.glide_critic(critic_observations)
        push_value = self.push_critic(critic_observations)
        reg_value = self.reg_critic(critic_observations)
        return value, glide_value, push_value, reg_value


class SoftplusWithOffset(nn.Module):
    def __init__(self, beta=1, threshold=20):
        super(SoftplusWithOffset, self).__init__()
        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        return self.softplus(x) + 1 + 1e-6
    
class StraightThroughSoftmax(nn.Module):
    def __init__(self):
        super(StraightThroughSoftmax, self).__init__()

    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)
        sampled_index = torch.multinomial(probs, 1)
        one_hot = torch.zeros_like(probs).scatter_(1, sampled_index, 1)
        return (one_hot - probs).detach() + probs, probs

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "softmax":
        return nn.Softmax()
    elif act_name == "softplusOffset":
        return SoftplusWithOffset()
    elif act_name == "straightThroughSoftmax":
        return StraightThroughSoftmax()
    else:
        print("invalid activation function!")
        return None