import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout,Input

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error


# Loading Data and Train test Split (80,20)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='./data',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    seed=7,
    validation_split=0.2,
    subset='training'
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='./data',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
    seed=7,
    validation_split=0.2,
    subset='validation'
)


class_names = train_ds.class_names
print("Class names:", class_names)

for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

# Normalize
def process(image,label):
    return tf.cast(image / 255., tf.float32), label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# create CNN model

model = Sequential()
model.add(Input(shape=(256, 256, 3)))
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

#Compile
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Train
history = model.fit(train_ds,epochs=30,validation_data=validation_ds,verbose=1)


# Plot Accuracy
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()

#Save Model
model.save("face_mask_model.keras")
print("Completed")