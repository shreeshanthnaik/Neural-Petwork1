import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Define Constants ---
BASE_DIR = 'dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'validation')

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# --- 2. Create Data Generators ---
print("Setting up data generators...")
# Training Data Generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation Data Generator (only normalize)
validation_datagen = ImageDataGenerator(rescale=1./255)

# --- 3. Load Images from Directories ---
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

# --- (Optional) Visualize a few augmented images ---
print("Showing a batch of augmented images (close this window to continue)...")
img_batch, label_batch = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img_batch[i])
    if label_batch[i] == 0:
        plt.title("Cat")
    else:
        plt.title("Dog")
    plt.axis("off")
plt.suptitle("Sample Augmented Training Images")
plt.show() # <-- The script will PAUSE here until you close the plot window


# --- 4. Build the CNN Model ---
print("Building the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    
    Dense(512, activation='relu'),
    Dropout(0.5), # Helps prevent overfitting
    Dense(1, activation='sigmoid') # Sigmoid for binary (0 or 1) output
])

# Show the model architecture
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
# This will take a few minutes!
history = model.fit(
    train_generator,
    # Calculate steps per epoch
    steps_per_epoch= train_generator.n // BATCH_SIZE, 
    epochs=15, # You can start with 15 and see how it does
    validation_data=validation_generator,
    # Calculate validation steps
    validation_steps= validation_generator.n // BATCH_SIZE
)

print("Training finished!")

# --- 7. Save the Model ---
# This is the file we will use in our web app
model_filename = 'cats_vs_dogs.h5'
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

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show() # <-- A new plot window will appear

print("All done!")