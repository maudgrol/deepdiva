import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

# i use a dataset train = 10000, test=1000, only 4 knobs randomized.

PATH = "../small_data280/"

# leave these as they are if you didn't change the simple_mfcc_datagenerator
FEATURES = f"{PATH}train_0_features.npy"
TARGET = f"{PATH}train_0_patches.npy"
TEST_FEATURES = f"{PATH}test_0_features.npy"
TEST_TARGET = f"{PATH}test_0_patches.npy"

mfcc_np = np.load(FEATURES)
X = np.expand_dims(mfcc_np, axis=-1)
y = np.load(TARGET)
X_test = np.expand_dims(np.load(TEST_FEATURES), axis=-1)
y_test = np.load(TEST_TARGET)
print(X.shape)

model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation="relu", padding="same", input_shape=(X.shape[1], X.shape[2], 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(8, activation="linear"))

model.summary()

model.compile(
    optimizer="adam",
    loss="mean_squared_error",  # euclidean_distance_loss
    metrics=["cosine_similarity"]
)

history = model.fit(X, y,
                    epochs=200,
                    batch_size=256,
                    validation_data=(X_test, y_test)
                    )

model.save(f'{PATH}model_linear')

plt.plot(history.history["cosine_similarity"], label="cosine_similarity")
plt.plot(history.history["val_cosine_similarity"], label="val_cosine_similarity")

plt.legend()
plt.show()
plt.close()
