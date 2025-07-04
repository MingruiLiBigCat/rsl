# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime
import tr_env_gym
#import isaacgym
#from legged_gym.envs import *
#from legged_gym.utils import get_args, task_registry
from on_policy_runner import OnPolicyRunner
import torch
#import wandb

def train(args=None):
    # args.headless = False
    # log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    # try:
    #     os.makedirs(log_pth)
    # except:
    #     pass

    # mode = "disabled"
    # if args.debug:
    #     args.headless = False
    #     mode = "disabled"
    #     args.rows = 10
    #     args.cols = 8
    #     args.num_envs = 10
    # # if args.wandb:
    # #     mode = "online"
    # print(args.proj_name)
    # wandb.init(project=args.proj_name, name=args.exptid, group=args.exptid[:3], mode=mode, dir="../../logs")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")

    #env, env_cfg = task_registry.make_env(name=args.task, args=args)
    #ppo_runner, train_cfg = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)
    train_cfg_dict = {
    "runner": {
        "algorithm_class_name": "PPO",
        "num_steps_per_env": 64,
        "save_interval": 100,
        "max_iterations": 10**5  
    },
    "algorithm": {
        "num_learning_epochs": 5,
        "num_mini_batches": 8,
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.001,
        "learning_rate": 3e-4,
        "clip_param": 0.2,
        "value_loss_coef": 0.5
    },
    "policy": {
        "activation": "relu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 0.5
    }
    }
    log_dir="./logs"
    env = tr_env_gym.tr_env_gym(render_mode="None",
                                    xml_file=os.path.join(os.getcwd(),"3prism_jonathan_steady_side.xml"),
                                    robot_type="j",
                                    is_test = False,
                                    desired_action = "straight",
                                    desired_direction = 1
                                    )
    env.reset_model()
    runner = OnPolicyRunner(env, 
                                train_cfg_dict, 
                                log_dir, 
                                init_wandb=False,
                                device="cuda")
    runner.learn(num_learning_iterations=10**5)

if __name__ == '__main__':
    # Log configs immediately
    train( )
