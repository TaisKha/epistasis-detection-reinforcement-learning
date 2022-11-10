import gym
import ptan
import numpy as np
import argparse
import time
import sys
import wandb


import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from EpiEnv import EpiEnv


GAMMA = 0.99
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.01
BATCH_SIZE = 32
NUM_ENVS = 1
CUDA = False
REWARD_STEPS = 4
CLIP_GRAD = 0.1

class RewardTracker:
    def __init__(self, stop_reward):
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        wandb.finish()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        sys.stdout.flush()
        if epsilon is not None:
            wandb.log({}, commit=False)
        wandb.log({"speed": speed, "reward_100": mean_reward, "reward": reward, "frame": frame}, commit=True)
        if reward >= self.stop_reward:
            f = open(LOG_FILE, 'a')
            f.write("Solved in %d frames!\n" % frame)
            f.close()
            return True
        return False


class NetA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NetA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    actions_t = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v

class EpiProbabilityActionSelector(ptan.actions.ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        assert isinstance(probs[0], np.ndarray)
        actions = []

        for prob in probs:
            num_selected_snps = NUM_SNPS
            chosen_snp = np.random.choice(len(prob), size=num_selected_snps, replace=False, p=prob)
            action = np.zeros(len(prob))
            for snp in chosen_snp:
                action[snp] = 1
            actions.append(action)
        return np.array(actions)
    
if __name__ == "__main__":
    
    device = torch.device("cuda" if CUDA else "cpu")

    
    for experiment in range(NUM_OF_EXPERIMENTS):
        make_env = lambda: EpiEnv()
        envs = [make_env() for _ in range(NUM_ENVS)]

        net = NetA2C(envs[0].observation_space.shape, envs[0].N_SNPS).to(device)
        print(net)

        agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device, action_selector=EpiProbabilityActionSelector())
        exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

        batch = []
        
        wandb.init(project="epistasis", entity="taisikus", config={
              "learning_rate": LEARNING_RATE,
              "gamma": GAMMA,
              "episodes_to_train": BATCH_SIZE,
              "sample_size": SAMPLE_SIZE,
              "reward_steps": REWARD_STEPS,
              "entropy_beta": ENTROPY_BETA,
              "episode_length": EPISODE_LENGTH
            })
        f = open(LOG_FILE, 'a')
        f.write("Experiment number " + str(experiment) + "\n")
        f.close()
        START = time.time()

        with RewardTracker(stop_reward=100) as tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if step_idx > MAX_ITER:
                    f = open(LOG_FILE, 'a')
                    END = time.time()
                    passed = END - START
                    f.write(f"Maximum iter is reached. Time passed : {passed}\n")
                    passed = END - START
                    f.close()
                    break
                    
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * torch.diagonal(torch.mm(log_prob_v, torch.transpose(actions_t, 0, 1)))
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                wandb.log({"advantage": adv_v,
                          "values": value_v,
                          "batch_rewards": vals_ref_v,
                          "loss_entropy": entropy_loss_v,
                          "loss_policy": loss_policy_v,
                          "loss_value": loss_value_v,
                          "loss_total": loss_v,
                          "grad_l2": np.sqrt(np.mean(np.square(grads))),
                          "grad_max": np.max(np.abs(grads)),
                          "grad_var": np.var(grads),
                          "step_idx": step_idx
                          })
