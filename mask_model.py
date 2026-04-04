import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# dataset path
DIRECTORY = "data"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

# load images
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        data.append(image)
        labels.append(category)

# convert data
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# encode labels
labels = np.where(labels == "with_mask", 1, 0)
labels = to_categorical(labels)

# split data
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# load base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(224, 224, 3))

# build head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# compile
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=1e-4),
              metrics=["accuracy"])

print("Training model...")

# train
model.fit(trainX, trainY, batch_size=32, epochs=5,
          validation_data=(testX, testY))

# save model
model.save("mask_model.h5")

print("Model saved successfully")