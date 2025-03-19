import os
import pickle
import tensorflow as tf
from typing import Dict, Any

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, inputs):
        e = tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        return tf.reduce_sum(context, axis=1)

def load_models(models_dir: str = "models") -> Dict[str, Any]:
    models = {}
    for file in os.listdir(models_dir):
        if file.endswith(".keras"):
            try:
                ticker = file.split("_")[0].upper()
                model_path = os.path.join(models_dir, file)
                
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={"AttentionLayer": AttentionLayer}
                )
                
                scaler_X = pickle.load(
                    open(os.path.join(models_dir, f"{ticker}_scaler_X.pkl"), "rb")
                )
                scaler_y = pickle.load(
                    open(os.path.join(models_dir, f"{ticker}_scaler_y.pkl"), "rb")
                )
                
                models[ticker] = {
                    "model": model,
                    "scaler_X": scaler_X,
                    "scaler_y": scaler_y
                }
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
    return models