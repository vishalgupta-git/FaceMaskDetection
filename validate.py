import tensorflow as tf
from tensorflow import keras
import os

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the saved model
model = keras.models.load_model("face_mask_model.keras")
print("Model loaded successfully!")

# Load validation dataset
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

# Normalize the images
def process(image, label):
    return tf.cast(image / 255., tf.float32), label

validation_ds = validation_ds.map(process)

# Evaluate the model
loss, accuracy = model.evaluate(validation_ds, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
