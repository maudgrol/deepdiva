import numpy as np
import pickle

NAME_OF_DATASET = "batch1"
DATA_PATH = "../data/data_tiny4/"
START= 0
SIZE = 81
USE_EXISTING_SCALER=False

all_patches = np.concatenate([np.load(f"{DATA_PATH}{i}_patches.npy") for i in range(START, SIZE)], axis=0)
all_mfcc = np.concatenate([np.load(f"{DATA_PATH}{i}_mfcc.npy") for i in range(START, SIZE)], axis=0)

##scaling the mfcc and saving the min and max values used

if USE_EXISTING_SCALER:
    f = open(f'{DATA_PATH}scaler.pckl', 'rb')
    scaler = pickle.load(f)
    print(scaler)
    print(scaler.shape)
    max = scaler[1]
    min = scaler[0]
    f.close()
    all_mfcc_normalized = (all_mfcc-min) / (max-min)
else:
    max = np.max(all_mfcc)
    min = np.min(all_mfcc)
    all_mfcc_normalized = (all_mfcc-min) / (max-min)
    scaler = np.array([min, max])
    f = open(f'{DATA_PATH}scaler.pckl', 'wb')
    pickle.dump(scaler, f)
    f.close()




print(all_patches.shape)
print(all_mfcc.shape)
np.save(f"{DATA_PATH}/{NAME_OF_DATASET}_patches.npy", all_patches)
np.save(f"{DATA_PATH}/{NAME_OF_DATASET}_mfcc.npy", all_mfcc_normalized)

