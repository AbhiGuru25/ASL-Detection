import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cells = [
    "# American Sign Language Detection",
    "## Objective\nBuild a system that can detect a given ASL input image and output what the sign represents (what letter of the alphabet is the sign).",
    "## Setup and Download Dataset\nPlease ensure you have your `kaggle.json` uploaded to your environment or configured correctly if running locally.",
    "## Import Libraries",
    "## Data Loading and Preprocessing",
    "## Model Building",
    "## Training the Model",
    "## Evaluation",
    "## Saving Model"
]

code_cells = [
    # cell 0
    """!pip install kaggle
import os
import shutil
# Automatically download dataset if kaggle.json is present in ~/.kaggle
# Make sure to place kaggle.json in ~/.kaggle/kaggle.json
os.makedirs('data', exist_ok=True)
!kaggle datasets download -d grassknoted/asl-alphabet -p data/
!unzip -q data/asl-alphabet.zip -d data/
print("Dataset downloaded and extracted.")""",
    
    # cell 1
    """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(tf.__version__)""",
    
    # cell 2
    """train_dir = 'data/asl_alphabet_train/asl_alphabet_train'
test_dir = 'data/asl_alphabet_test/asl_alphabet_test'

img_size = 64
batch_size = 64

# Data Augmentation and Generator for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2 # use 20% for validation
)

print("Training data:")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

print("Validation data:")
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
""",
    
    # cell 3
    """num_classes = train_generator.num_classes
print(f"Number of classes: {num_classes}")

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
""",

    # cell 4
    """early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

epochs = 15 # Set to a higher value for better accuracy

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)""",

    # cell 5
    """plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Evaluation on validation set
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
""",
    
    # cell 6
    """model.save('asl_model.h5')
print("Model saved as asl_model.h5")"""
]

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[1]),
    nbf.v4.new_markdown_cell(text_cells[2]),
    nbf.v4.new_code_cell(code_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[3]),
    nbf.v4.new_code_cell(code_cells[1]),
    nbf.v4.new_markdown_cell(text_cells[4]),
    nbf.v4.new_code_cell(code_cells[2]),
    nbf.v4.new_markdown_cell(text_cells[5]),
    nbf.v4.new_code_cell(code_cells[3]),
    nbf.v4.new_markdown_cell(text_cells[6]),
    nbf.v4.new_code_cell(code_cells[4]),
    nbf.v4.new_markdown_cell(text_cells[7]),
    nbf.v4.new_code_cell(code_cells[5]),
    nbf.v4.new_markdown_cell(text_cells[8]),
    nbf.v4.new_code_cell(code_cells[6]),
]

with open('ASL_Detection.ipynb', 'w') as f:
    nbf.write(nb, f)
