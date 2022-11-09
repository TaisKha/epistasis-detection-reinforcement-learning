import numpy as np
import gym
from gym import spaces
import json
from collections import defaultdict
import ptan
import os
from typing import Optional

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from environment import EpistasisEnv

EPISODE_LENGTH = 1
NUM_SNPS = 3
SAMPLE_SIZE = 600

GAMMA = 0.99
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
WANDB = True
REWARD_STEPS = 4
ENTROPY_BETA = 0.01
MAX_STEPS = 5000
NUMBER_OF_EXPERIMENTS = 50

    
class FixedEpistasisEnv(gym.Env):

    def __init__(self, sample_size, n_snps, observation_onehot, filename, observation, obs_phenotypes, disease_snps):
        self.one_hot_obs = observation_onehot
        self.filename = filename
        self.obs = observation
        self.obs_phenotypes = obs_phenotypes
        self.disease_snps = disease_snps
        
        self.SAMPLE_SIZE = sample_size #t1 = t2 = SAMPLE_SIZE
        self.N_SNPS = n_snps
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.N_SNPS,), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (3, 2*self.SAMPLE_SIZE, self.N_SNPS), dtype=np.uint8)
        self.engine = None
        
        
    def normalize_reward(self, current_reward):
        maximum_env_reward = self._count_reward(self.disease_snps, check_on_true_answer=False)
        minimal_reward = 0.5
        
        if maximum_env_reward < current_reward:
            print("maximum_env_reward < current_reward", "\n current reward = ", current_reward, "\n maximum_env_reward = ", maximum_env_reward )
            current_reward *= 0.9
        normalized_reward = (current_reward - minimal_reward) / (maximum_env_reward - minimal_reward) 
        
        return normalized_reward

    
    def step(self, action):
        snp_ids = self._take_action(action)
        reward = self._count_reward(snp_ids)
# c нормализацией
        reward = self.normalize_reward(reward)
        reward *= 100
        self.current_step += 1
        done = self.current_step == EPISODE_LENGTH
        return self.one_hot_obs, reward, done, {}
    
    def _count_reward(self, snp_ids, check_on_true_answer=True):
        
        if set(snp_ids) == set(self.disease_snps) and check_on_true_answer:
            f = open('LOG_FILE', 'a')
            f.write("Disease snps are found ")
            f.write(str(snp_ids))
            f.write("\n")
            
            END = time.time()
            passed = END - START
            f.write(f" time passed: {passed} \n")
            f.close()
        
        all_existing_seq = defaultdict(lambda: {'control' : 0, 'case' : 0})
        for i, idv in enumerate(self.obs):
            snp_to_cmp = tuple(idv[snp_id] for snp_id in snp_ids) #tuple of SNP that 
            if self.obs_phenotypes[i] == 0:
                all_existing_seq[snp_to_cmp]['control'] += 1
            else:
                all_existing_seq[snp_to_cmp]['case'] += 1

        #count reward      
        TP = 0 #HR case
        FP = 0 #HR control
        TN = 0 #LR control
        FN = 0 #LR case

        for case_control_count in all_existing_seq.values():
          # if seq is in LR group
            if case_control_count['case'] <= case_control_count['control']: #вопрос <= или <
                FN += case_control_count['case']
                TN += case_control_count['control']
            else:
          # if seq is in HR group
                TP += case_control_count['case']
                FP += case_control_count['control']
        R = (FP + TN) / (TP + FN)
        delta = FP / (TP+0.001)
        gamma = (TP + FP + TN + FN) / (TP+0.001)
        CCR = 0.5 * (TP / (TP + FN) + TN / (FP + TN))
        U = (R - delta)**2 / ((1 + delta) * (gamma - delta - 1 + 0.001))
        return CCR + U

  
    def reset(self):
       
        self.current_step = 0
        
        return self.one_hot_obs

    def render(self, mode='human', close=False):
        pass
    
    def _take_action(self, action):
        chosen_snp_ids = []
        for i, choice in enumerate(action):
            if choice == 1:
                chosen_snp_ids.append(i)
        return chosen_snp_ids  

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

class PolicyAgent(ptan.agent.BaseAgent):

    def __init__(self, model, action_selector=EpiProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=ptan.agent.default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1, dtype=torch.double)
        # print(f"{probs_v.dtype}")    
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states

class SnpPGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SnpPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.double()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)  
    



def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val

if __name__ == "__main__":
    
    for experiment in range(NUMBER_OF_EXPERIMENTS):
        f = open('resultPG.txt', 'a')
        f.write('\n' + str(experiment) + ' experiment' + '\n')
        f.close()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        temp_env = EpistasisEnv()
        fixed_observation_onehot = temp_env.reset()
        fixed_filename = temp_env.filename
        fixed_observation = temp_env.obs
        fixed_obs_phenotypes = temp_env.obs_phenotypes
        fixed_disease_snps = temp_env.disease_snps
        fixed_sample_size = temp_env.SAMPLE_SIZE
        fixed_n_snps = temp_env.N_SNPS
        env = FixedEpistasisEnv(fixed_sample_size, fixed_n_snps, fixed_observation_onehot, fixed_filename, fixed_observation, fixed_obs_phenotypes, fixed_disease_snps)

        if WANDB:
            wandb.init(project="epistasis", entity="taisikus", config={
              "learning_rate": LEARNING_RATE,
              "gamma": GAMMA,
              "episodes_to_train": BATCH_SIZE,
              "sample_size": SAMPLE_SIZE,
              "reward_steps": REWARD_STEPS,
              "entropy_beta": ENTROPY_BETA,
              "episode_length": EPISODE_LENGTH
            })

        net = SnpPGN(env.observation_space.shape, env.N_SNPS)
        net = nn.DataParallel(net)
        net.double()
        net.to(device)
        print(net)
        agent = PolicyAgent(net, action_selector=EpiProbabilityActionSelector(), apply_softmax=True, device=device)
        exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        
        total_rewards = []
        step_rewards = []
        step_idx = 0
        done_episodes = 0
        reward_sum = 0.0

        bs_smoothed = entropy = l_entropy = l_policy = l_total = None
        batch_states, batch_actions, batch_scales = [], [], []

        START = time.time()

        for step_idx, exp in enumerate(exp_source):
            if step_idx > MAX_STEPS and WANDB:
                break
                wandb.finish() 

            reward_sum += exp.reward
            baseline = reward_sum / (step_idx + 1)
            if WANDB:
                wandb.log({'baseline': baseline, 'step_idx': step_idx}, commit=False)

            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            batch_scales.append(exp.reward - baseline)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                done_episodes += 1
                reward = new_rewards[0]
                total_rewards.append(reward)
                mean_rewards = float(np.mean(total_rewards[-100:]))
                if WANDB:
                    wandb.log({'reward':reward, 'reward_100':mean_rewards, 'episodes':done_episodes}, commit=False)

            if len(batch_states) < BATCH_SIZE:
                continue

            optimizer.zero_grad()

            states_v = torch.stack(batch_states)
            states_v = states_v.to(device)
            batch_actions_t = torch.DoubleTensor(batch_actions)
            batch_actions_t = batch_actions_t.to(device)
            batch_scale_v = torch.DoubleTensor(batch_scales)
            batch_scale_v = batch_scale_v.to(device)

            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_scale_v * torch.diagonal(torch.mm(log_prob_v, torch.transpose(batch_actions_t, 0, 1)))
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(logits_v, dim=1)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v

            loss_v.backward()
            optimizer.step()

            # calc KL-div
            new_logits_v = net(states_v)
            new_prob_v = F.softmax(new_logits_v, dim=1)
            kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
            wandb.log({'kl':kl_div_v.item()}, commit=False)

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                grad_max = max(grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1

            bs_smoothed = smooth(bs_smoothed, np.mean(batch_scales))
            entropy = smooth(entropy, entropy_v.item())
            l_entropy = smooth(l_entropy, entropy_loss_v.item())
            l_policy = smooth(l_policy, loss_policy_v.item())
            l_total = smooth(l_total, loss_v.item())
            if WANDB:
                wandb.log({"entropy":entropy, "loss_entropy": l_entropy, "loss_policy": l_policy, "loss_total": l_total, "grad_l2": grad_means / grad_count, "grad_max": grad_max, "batch_scales": bs_smoothed}, commit=True)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()
