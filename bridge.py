import numpy as np
import pandas as pd
import os
import warnings

ref = 'aba_bach_bank'  
dirs = ['drug_to_german', 'to_end']

i = 0
# Read from aba_bach_bank 
path = os.path.join('tmp', ref)
for file in os.listdir(path):
    files = []
    files.append(os.path.join(path, file))
    print(f"Processing file: {files[0]}")
    if file.startswith('experiment_'):
        # Search same file name in dirs
        for d in dirs:
            for f in os.listdir(os.path.join('tmp', d)):
                if f == file:
                    files.append(os.path.join('tmp', d, f))
                    
        final_df = pd.DataFrame()
        for f in files:
            print(f"Final df shape before merging: {final_df.shape}")
            print(f"Reading file: {f}")
            # merge data from files
            data = pd.read_csv(f)
            print(f"Data shape from file {f}: {data.shape}")
            final_df = pd.concat([final_df, data], axis=0)
        
        # Save final dataframe
        output_path = os.path.join('tmp', 'final', file)
        final_df.to_csv(output_path, index=False)
    
    print(f"Completed processing for file: {file}")
            
