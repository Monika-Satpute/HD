import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
data_dir = 'PlantVillage'  # Make sure this folder is uploaded
img_size = (128, 128)

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Show class names
print("Class indices:", train_data.class_indices)

# Show head sample
sample_images, sample_labels = next(train_data)
print("Sample shape:", sample_images[0].shape)

# Plot sample images
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(sample_images[i])
    plt.title(np.argmax(sample_labels[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train
history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[early_stop])

# Save model
model.save('plant_disease_model.h5')

# Evaluate on a test sample
test_sample = sample_images[0:1]
pred = model.predict(test_sample)
print("Predicted class:", np.argmax(pred))

# https://www.kaggle.com/datasets/emmarex/plantdisease
