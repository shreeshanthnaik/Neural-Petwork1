import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Define Constants ---
BASE_DIR = 'dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'validation')

# We use 224x224, the standard size for MobileNetV2
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. Create Data Generators ---
# NOTE: We are NOT rescaling here (no rescale=1./255)
# The MobileNetV2 preprocessing is built into the model
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator() # No augmentation for validation

# --- 3. Load Images ---
# We use the generators to load images
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# --- 4. Build the Transfer Learning Model ---
print("Building the Transfer Learning model...")

# Define the input layer
inputs = Input(shape=(224, 224, 3))

# --- Preprocessing Layer ---
# This layer handles the normalization (scaling pixels from 0-255 to -1 to 1)
# which is what MobileNetV2 expects.
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
x = preprocess_input(inputs)

# --- Base Model (MobileNetV2) ---
# Load the MobileNetV2 model, pre-trained on 'imagenet'
# include_top=False means we DON'T want its original final classifier layer
base_model = MobileNetV2(
    input_tensor=x,
    include_top=False,
    weights='imagenet'
)

# Freeze the base model. We don't want to re-train its weights.
base_model.trainable = False
print("Base model (MobileNetV2) loaded and frozen.")

# --- Our Custom "Head" ---
# Add our new custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Averages all the features
x = Dropout(0.5)(x) # Helps prevent overfitting
outputs = Dense(1, activation='sigmoid')(x) # Our final prediction layer

# Create the new model
model = Model(inputs, outputs)

# Show the new model architecture
model.summary()

# --- 5. Compile the Model ---
print("Compiling the model...")
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# --- 6. Train the Model ---
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=10, # 10 epochs is often enough for transfer learning
    validation_data=validation_generator,
    validation_steps=validation_generator.n // BATCH_SIZE
)

print("Training finished!")

# --- 7. Save the New Model ---
model_filename = 'cats_vs_dogs_v2.h5'
model.save(model_filename)
print(f"Model has been saved as {model_filename}")

# --- 8. Plot Training Results ---
print("Plotting training history...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

print("All done!")