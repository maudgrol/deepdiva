import os
import numpy as np

DATA_PATH = "../data/"

# Load data files
trainFeatures = np.load(os.path.join(DATA_PATH, "toy_data/train_0_features.npy"))
trainParams = np.load(os.path.join(DATA_PATH, "toy_data/train_0_patches.npy"))
testFeatures = np.load(os.path.join(DATA_PATH, "toy_data/test_0_features.npy"))
testParams = np.load(os.path.join(DATA_PATH, "toy_data/test_0_patches.npy"))

trainMels = np.load(os.path.join(DATA_PATH, "toy_data/train_0_melspectrogram.npy"))

# Check dimensions of training and test data
print(f"The shape of trainFeatures: {trainFeatures.shape}")
print(f"The shape of trainParams: {trainParams.shape}")
print(f"The shape of testFeatures: {testFeatures.shape}")
print(f"The shape of testParams: {testParams.shape}")
print(f"The shape of trainMels: {trainMels.shape}")
