import multiprocessing  # the module we will be using for multiprocessing
import os
import numpy as np

def work(idx):
    
    filename = f"./epigen/sim/{idx}_1_ASW.json"
    if not os.path.exists(filename):
        command = ['cd ./epigen/ && python3']
        os.system(f" simulate_data.py --sim-ids {idx} --corpus-id 1 --pop ASW --inds 5000 --snps 100 --model models/ext_model.ini")
    

    
if __name__ == "__main__":  # Allows for the safe importing of the main module
    print("There are %d CPUs on this machine" % multiprocessing.cpu_count())
    number_processes = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(number_processes)
    total_tasks = 1000
    tasks = range(total_tasks)
    results = pool.map_async(work, tasks)
    pool.close()
    pool.join()
