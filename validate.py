import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

# 1. Load the saved model
model = keras.models.load_model("face_mask_model.keras")
print("Model loaded successfully!")

# 2. Load validation dataset
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

# Store class names (e.g., ['with_mask', 'without_mask'])
class_names = validation_ds.class_names

# 3. Normalize the images
def process(image, label):
    return tf.cast(image / 255., tf.float32), label

validation_ds = validation_ds.map(process)

# 4. Evaluate the model (Quick Check)
loss, accuracy = model.evaluate(validation_ds, verbose=1)
print(f"\nOverall Validation Loss: {loss:.4f}")
print(f"Overall Validation Accuracy: {accuracy:.4f}\n")

# 5. Collect True Labels and Predictions
y_true = []
y_pred = []

print("Extracting predictions for Confusion Matrix...")
for images, labels in validation_ds:
    preds = model.predict(images, verbose=0)
    
    if preds.shape[-1] == 1:
        y_pred.extend((preds > 0.5).astype("int32").flatten())
    else:
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true.extend(labels.numpy())

# 6. Generate Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 7. Generate and Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix: Face Mask Detection')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the plot
plt.savefig('assets/confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()