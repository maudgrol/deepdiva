import numpy as np
import matplotlib.pyplot as plt

#i use a dataset train = 10000, test=1000, only 4 knobs randomized.

PATH = "../small_data3/"
FEATURES = f"{PATH}train_0_features.npy"
TARGET = f"{PATH}train_0_patches.npy"
TEST_FEATURES  = f"{PATH}test_0_features.npy"
TEST_TARGET = f"{PATH}test_0_patches.npy"


mfcc_np = np.load(FEATURES)
X = np.expand_dims(mfcc_np, axis=-1)
y = np.load(TARGET)
X_test = np.expand_dims(np.load(TEST_FEATURES), axis=-1)
y_test = np.load(TEST_TARGET)
print(X.shape)


from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation="relu", padding="same", input_shape=(44, 13, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(4, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(4, activation="sigmoid"))

model.summary()


model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["accuracy"]
)

history = model.fit( X, y,
    epochs=200, validation_data=(X_test, y_test)
    )

model.save(f'{PATH}model')

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")

plt.legend()
plt.show()
plt.close()