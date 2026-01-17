import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input, Rescaling, RandomFlip, RandomRotation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import kaggle
import matplotlib.pyplot as plt

if tf.config.list_physical_devices('GPU'):
    print("GPU Available: Training will be fast.")
else:
    print("GPU NOT Available - Check drivers/CUDA.")
    print(tf.config.list_physical_devices())
    choice = input("Do you wish to continue? It will be significantly slower [Y/N]: ").strip().upper()
    if choice != 'Y':
        print("Exiting...")
        exit()
        

#Download the Face Image Dataset
kaggle.api.authenticate()
kaggle.api.dataset_download_files('omkargurav/face-mask-dataset',unzip=True)

# --- 2. Load Data ---
BATCH_SIZE = 32
IMG_SIZE = (256, 256)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='./data',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    seed=7,
    validation_split=0.2,
    subset='training'
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='./data',
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    seed=7,
    validation_split=0.2,
    subset='validation'
)

# --- 3. Performance Optimization (Caching & Prefetching) ---
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    # Normalize images to [0, 1]
    return tf.cast(image / 255., tf.float32), label

train_ds = train_ds.map(preprocess).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.map(preprocess).cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Define Data Augmentation ---
data_augmentation = Sequential([
  RandomFlip("horizontal"),
  RandomRotation(0.1),
])

# --- 5. Improved CNN Architecture ---
model = Sequential()
model.add(Input(shape=(256, 256, 3)))

# Apply augmentation only during training
model.add(data_augmentation)

# Block 1
model.add(Conv2D(32, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

# Block 2
model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

# Block 3
model.add(Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))

model.add(Flatten())

# Dense Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # Increased Dropout to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) # Increased Dropout
model.add(Dense(1, activation='sigmoid'))

# --- 6. Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('face_mask_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# --- 7. Train ---
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=validation_ds,
    callbacks=callbacks, # Added callbacks here
    verbose=1
)

# --- 8. Plotting ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.title('Loss')
plt.legend()
plt.show()

print("Completed")