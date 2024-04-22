# %%
import os

import pickle

from Gen_syn_data import DataGenerator

# %%

FILE_DIR = os.getcwd()

METADATA = {
    'Total_Dimensions': [10,100,1000], # total dimensions (DAG and random)
    'N': [50,500,5000],
    'random_features_fraction': 0.4 # what fraction of total will be random
}

METADATA = {
    'Total_Dimensions': [10,100,1000], # total dimensions (DAG and random)
    'N': [50],
    'random_features_fraction': 0.4 # what fraction of total will be random
}

# %%
# Generate all dimensions for all experiments
Ns = [] # Number of instances
num_random = [] # Number of random features
p = [] # Number of features in the DAG (relevant+irrelevant)

for total_dim in METADATA['Total_Dimensions']:
    for n in METADATA['N']:
        Ns.append(n)
        num_random.append(int(total_dim*METADATA['random_features_fraction']))
        p.append(int(total_dim-num_random[-1]))
        
        # Quality check
        assert total_dim == p[-1]+num_random[-1], 'Dimensions do not make sense'
# %%
# Generate synthetic data
for i in range(len(p)):
    print(f'Generating dataset ... N={Ns[i]}_p={p[i]}_d_random={num_random[i]}')
    datagen = DataGenerator(
        p=p[i],
        random_dim=num_random[i],
        n_branches=2
    )
    
    df, dag_info = datagen.generate_data(n=Ns[i], noise=0.005, threshold=0.5)
    
    save_path = os.path.join(FILE_DIR,'Datasets',f'N={Ns[i]}_p={p[i]}_d_random={num_random[i]}')
    os.makedirs(save_path,exist_ok=True)
    
    # Save data
    df.to_pickle(os.path.join(save_path,'data.pkl'))
    # Save dag_info as a JSON file
    with open(os.path.join(save_path, 'dag_info.pkl'), 'wb') as f:
        pickle.dump(dag_info, f)
    

# %%
