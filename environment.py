import gym

#имя файла с данными
FILENAME = "/home/tskhakharova/epistasis-rl/epigen/sim/100502_1_ASW.json"
#имя файла для output
LOG_FILE = "logs"

class EpistasisEnv(gym.Env):

    def __init__(self):
        self.SAMPLE_SIZE = SAMPLE_SIZE #t1 = t2 = SAMPLE_SIZE
        self.reset()
        self.action_space = spaces.Box(low=0, high=1, shape=(self.N_SNPS,), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (3, 2*self.SAMPLE_SIZE, self.N_SNPS), dtype=np.uint8)
        
        
    def establish_phen_gen(self, file):
        with open(file) as f:
            data = json.load(f)
            genotype = np.array(data["genotype"])
            self.phenotype = np.array(data["phenotype"])
            self.genotype = genotype.T
            num_phenotypes = max(self.phenotype)+1
            self.disease_snps = data["disease_snps"]
            self.phen_gen = [[] for _ in range(num_phenotypes)]
            for i in range(len(self.genotype)):
                self.phen_gen[self.phenotype[i]].append(i)  
            return  self.genotype.shape[0], self.genotype.shape[1]
        

    def normalize_reward(self, current_reward):
        maximum_env_reward = self._count_reward(self.disease_snps)
        minimal_reward = 0.5
        
        if maximum_env_reward < current_reward:
            print("maximum_env_reward < current_reward", "\n current reward = ", current_reward, "\n maximum_env_reward = ", maximum_env_reward )
            current_reward *= 0.9
        normalized_reward = (current_reward - minimal_reward) / (maximum_env_reward - minimal_reward) 
        
        return normalized_reward

    
    def step(self, action):
        snp_ids = self._take_action(action)
        reward = self._count_reward(snp_ids)
#         c нормализации
        reward = self.normalize_reward(reward)
        
        self.current_step += 1
        if self.current_step == EPISODE_LENGTH:
            done = True
        else:
            done = False  
        obs = None if done else self._next_observation()
        return obs, reward, done, {}
    
    def _count_reward(self, snp_ids):
        
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

#если необходимо использовать случайный файл

       # pops = ["ASW", "CEU", "CEU+TSI", "CHD", "GIH", "JPT+CHB", "LWK", "MEX", "MKK", "TSI"]
#        sim_idx = np.random.randint(2500)
#        corp_idx = np.random.randint(1, 23)
#        pop_idx = np.random.choice(pops)
#        self.filename = f"/home/tskhakharova/epistasis-rl/epigen/sim/{sim_idx}_{corp_idx}_{pop_idx}.json"
#        if not os.path.exists(self.filename):
#           os.system(f"cd /home/tskhakharova/epistasis-rl/epigen/ && python3 simulate_data.py --sim-ids {sim_idx} --corpus-id {corp_idx} --pop {pop_idx} --inds 5000 --snps 100 --model models/ext_model.ini")

#если используем фиксированный файл
        self.filename = FILENAME
        
        self.N_IDV, self.N_SNPS = self.establish_phen_gen(self.filename)
        self.obs_phenotypes = None
        one_hot_obs = self._next_observation()
        self.current_step = 0
        
        return one_hot_obs

    def render(self, mode='human', close=False):
        pass
    
    def _take_action(self, action):
        chosen_snp_ids = []
        for i, choice in enumerate(action):
            if choice == 1:
                chosen_snp_ids.append(i)
        return chosen_snp_ids    
    
    def _next_observation(self):
        id_0 = np.random.choice(self.phen_gen[0], self.SAMPLE_SIZE)
        id_1 = np.random.choice(self.phen_gen[1], self.SAMPLE_SIZE)
        sample_ids = np.array(list(zip(id_0,id_1))).flatten()
        self.obs = np.array([self.genotype[idv] for idv in sample_ids])
        self.obs_phenotypes = [self.phenotype[idv] for idv in sample_ids]
        #one_hot encoding
        one_hot_obs = F.one_hot(torch.tensor(self.obs), 3)
        one_hot_obs = one_hot_obs.movedim(2, 0)

        return one_hot_obs
    
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
