
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from .preprocessing import DEFAULT_IMAGE_SIZE, load_mammography_image
except ImportError:
    from preprocessing import DEFAULT_IMAGE_SIZE, load_mammography_image


def find_base_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(isinstance(child, tf.keras.layers.Conv2D) for child in layer.layers):
            return layer
    raise ValueError("Base convolucional nao encontrada para Grad-CAM.")


def find_last_conv_layer(base_model: tf.keras.Model) -> str:
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Camada convolucional nao encontrada para Grad-CAM.")


def call_layer(layer: tf.keras.layers.Layer, tensor: tf.Tensor) -> tf.Tensor:
    try:
        return layer(tensor, training=False)
    except TypeError:
        return layer(tensor)


def compute_grad_cam(
    model: tf.keras.Model,
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    input_tensor = tf.expand_dims(load_mammography_image(tf.constant(str(image_path)), image_size=image_size), axis=0)
    base_model = find_base_model(model)
    last_conv_name = find_last_conv_layer(base_model)
    last_conv_layer = base_model.get_layer(last_conv_name)
    feature_model = tf.keras.Model(base_model.inputs, [last_conv_layer.output, base_model.output])
    post_base_layers = model.layers[model.layers.index(base_model) + 1 :]

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(input_tensor)
        tape.watch(conv_outputs)
        prediction = features
        for layer in post_base_layers:
            prediction = call_layer(layer, prediction)
        malignant_score = prediction[:, 0]

    gradients = tape.gradient(malignant_score, conv_outputs)
    if gradients is None:
        raise ValueError("Nao foi possivel calcular gradientes para Grad-CAM.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_gradients[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) == 0.0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / max_value).numpy()


def unpad_heatmap(
    heatmap: np.ndarray,
    original_size: tuple[int, int],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    largura_original, altura_original = original_size
    altura_alvo, largura_alvo = image_size

    escala = min(altura_alvo / altura_original, largura_alvo / largura_original)
    altura_util = altura_original * escala
    largura_util = largura_original * escala

    offset_y = (altura_alvo - altura_util) / 2.0
    offset_x = (largura_alvo - largura_util) / 2.0

    ampliado = np.asarray(
        Image.fromarray(np.float32(heatmap)).resize(
            (largura_alvo, altura_alvo), Image.Resampling.BILINEAR
        )
    )

    inicio_y = int(round(offset_y))
    fim_y = int(round(offset_y + altura_util))
    inicio_x = int(round(offset_x))
    fim_x = int(round(offset_x + largura_util))

    recorte = ampliado[
        max(inicio_y, 0) : min(fim_y, altura_alvo),
        max(inicio_x, 0) : min(fim_x, largura_alvo),
    ]
    return recorte if recorte.size else ampliado


def overlay_heatmap(
    image_path: str | Path,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    original = Image.open(image_path).convert("RGB")
    alinhado = unpad_heatmap(heatmap, original.size, image_size=image_size)
    heatmap_image = Image.fromarray(np.uint8(alinhado * 255)).resize(original.size, Image.Resampling.BILINEAR)
    colored = plt.get_cmap("jet")(np.array(heatmap_image) / 255.0)[:, :, :3]
    colored_image = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(original, colored_image, alpha=alpha)
