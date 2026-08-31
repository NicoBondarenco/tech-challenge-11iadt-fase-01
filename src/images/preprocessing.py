
from __future__ import annotations

import tensorflow as tf


DEFAULT_IMAGE_SIZE = (224, 224)


def load_mammography_image(path: tf.Tensor, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize_with_pad(image, image_size[0], image_size[1])
    image = image * 255.0
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def conservative_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="conservative_augmentation",
    )


def make_dataset(
    image_paths: list[str],
    labels: list[int] | None = None,
    batch_size: int = 16,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    shuffle: bool = False,
    augment: bool = False,
) -> tf.data.Dataset:
    paths = tf.constant(image_paths)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(paths)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, tf.constant(labels, dtype=tf.float32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    augmenter = conservative_augmentation() if augment else None

    def _load(path: tf.Tensor, label: tf.Tensor | None = None):
        image = load_mammography_image(path, image_size=image_size)
        if augmenter is not None:
            image = augmenter(image, training=True)
        return (image, label) if label is not None else image

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
