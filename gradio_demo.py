import os
import io
import json
from typing import Tuple, Dict

import numpy as np
from PIL import Image
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model


class InceptionV3FeatureExtractor(tf.keras.layers.Layer):
    """Wrapper to load InceptionV3 for feature extraction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base_model_initialized = False

    def build(self, input_shape):
        if not self._base_model_initialized:
            self.base_model = tf.keras.applications.InceptionV3(
                weights="imagenet", include_top=False, input_shape=(224, 224, 3)
            )
            self.base_model.trainable = False
            self._base_model_initialized = True
        super().build(input_shape)

    def call(self, inputs):
        x = tf.image.resize(inputs, (224, 224))
        return self.base_model(x, training=False)


def load_custom_inception_model(model_path: str):
    """Load the trained InceptionV3 model with custom objects."""
    custom_objects = {"InceptionV3FeatureExtractor": InceptionV3FeatureExtractor}
    return load_model(model_path, custom_objects=custom_objects)


# Load model and class names
MODEL_PATH = "artifacts/fashion_mnist_inception.h5"
CLASS_FILE = "artifacts/class_names.json"

try:
    model = load_custom_inception_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

with open(CLASS_FILE, "r") as f:
    CLASS_NAMES = json.load(f)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized tensor."""
    if image.mode != "L":
        image = image.convert("L")
    image = np.array(image)
    image = tf.expand_dims(image, -1)
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()


def predict(image: Image.Image) -> Tuple[str, Dict[str, float]]:
    """Run model prediction and return class name and confidences."""
    processed = preprocess_image(image)
    preds = model.predict(np.expand_dims(processed, 0))[0]
    predicted_class = CLASS_NAMES[int(np.argmax(preds))]
    confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidences


def main():
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Textbox(label="Predicted Class"), gr.Label(num_top_classes=len(CLASS_NAMES))],
        title="Fashion MNIST Classifier",
        description="Upload a fashion image to classify it",
    )
    demo.launch()


if __name__ == "__main__":
    main()
