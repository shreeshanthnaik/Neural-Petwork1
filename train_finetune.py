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
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. Re-build the v2 Model Architecture ---
print("Building model architecture...")
inputs = Input(shape=(224, 224, 3))
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
x = preprocess_input(inputs)
base_model = MobileNetV2(
    input_tensor=x,
    include_top=False,
    weights=None 
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

# --- 3. Load the Trained Weights ---
print("Loading trained weights from cats_vs_dogs_v2.h5...")
model.load_weights('cats_vs_dogs_v2.h5')
print("Weights loaded successfully.")

# --- 4. Unfreeze the Top Layers of the Base ---
# --- FIX 1: Use the 'base_model' variable directly ---
base_model.trainable = True 

# --- FIX 2: Use 'base_model' variable ---
print(f"Total layers in base_model: {len(base_model.layers)}")
fine_tune_at = 100 

# --- FIX 3: Use 'base_model' variable ---
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print("Base model unfrozen for fine-tuning.")

# --- 5. Re-Compile with a Very Low Learning Rate ---
print("Re-compiling model with a low learning rate...")
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    metrics=['accuracy']
)

model.summary()

# --- 6. Set up Data Generators ---
print("Setting up data generators...")
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

# --- 7. Continue Training (Fine-Tuning) ---
print("Starting fine-tuning...")
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BATCH_SIZE,
    epochs=10, 
    validation_data=validation_generator,
    validation_steps=validation_generator.n // BATCH_SIZE
)

print("Fine-tuning finished!")

# --- 8. Save the Final Model ---
model_filename = 'cats_vs_dogs_v3_final.h5'
model.save(model_filename)
print(f"Model has been saved as {model_filename}")

# --- 9. Plot Fine-Tuning Results ---
print("Plotting fine-tuning history...")
acc = history_fine_tune.history['accuracy']
val_acc = history_fine_tune.history['val_accuracy']
loss = history_fine_tune.history['loss']
val_loss = history_fine_tune.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Fine-Tune Accuracy')
plt.plot(epochs, val_acc, 'b', label='Fine-Tune Validation Accuracy')
plt.title('Fine-Tuning Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Fine-Tune Loss')
plt.plot(epochs, val_loss, 'b', label='Fine-Tune Validation Loss')
plt.title('Fine-Tuning Loss')
plt.legend()
plt.show()

print("All done!")