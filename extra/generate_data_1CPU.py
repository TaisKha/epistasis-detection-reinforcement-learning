import os
import numpy as np

amount_of_data = 1000
# random_sim_idx = np.random.randint(self.AMOUNT_OF_DATA)
for idx in range(amount_of_data):
    filename = f"./epigen/sim/{idx}_1_ASW.json"
    if not os.path.exists(filename):
        os.system(f"cd ./epigen/ && python3 simulate_data.py --sim-ids {idx} --corpus-id 1 --pop ASW --inds 5000 --snps 100 --model models/ext_model.ini")
