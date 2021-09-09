import os
import spiegelib as spgl
import numpy as np

DATA_PATH = "../data/"

# Load data files
trainFeatures = np.load(os.path.join(DATA_PATH, "data_mfcc/train_features.npy"))
trainParams = np.load(os.path.join(DATA_PATH, "data_mfcc/train_patches.npy"))
testFeatures = np.load(os.path.join(DATA_PATH, "data_mfcc/test_features.npy"))
testParams = np.load(os.path.join(DATA_PATH, "data_mfcc/test_patches.npy"))

# Check dimensions of training and test data
print(f"The shape of trainFeatures: {trainFeatures.shape}")
print(f"The shape of trainParams: {trainParams.shape}")
print(f"The shape of testFeatures: {testFeatures.shape}")
print(f"The shape of testParams: {testParams.shape}")
