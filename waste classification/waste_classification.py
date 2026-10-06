import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Settings

dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset",
    "WasteSnap_Dataset"
)

img_height = 128
img_width = 128
batch_size = 32
epochs = 10

# 2. Load training data

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# 3. Load validation data

validation_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


# 4. Show class names

class_names = train_data.class_names

print("Classes:")
print(class_names)


# 5. Improve data loading

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

validation_data = validation_data.prefetch(
    buffer_size=AUTOTUNE
)


# 6. Create CNN model

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(img_height, img_width, 3)
    ),

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# 7. Compile the model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 8. Display model

model.summary()


# 9. Train the model

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs
)


# 10. Save the model

model.save(
    "waste_classification_model.keras"
)

print("Model saved successfully!")


# 11. Accuracy graph

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.savefig(
    "accuracy_graph.png"
)

plt.show()


# 12. Loss graph

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.savefig(
    "loss_graph.png"
)

plt.show()


# 13. Test an image

image_path = input(
    "Enter image path: "
)

image = tf.keras.utils.load_img(
    image_path,
    target_size=(img_height, img_width)
)

image_array = tf.keras.utils.img_to_array(
    image
)

image_array = tf.expand_dims(
    image_array,
    0
)


# 14. Make prediction

prediction = model.predict(
    image_array
)

predicted_class = class_names[
    np.argmax(prediction[0])
]

confidence = np.max(
    prediction[0]
) * 100


print()
print(
    "Predicted waste:",
    predicted_class
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)