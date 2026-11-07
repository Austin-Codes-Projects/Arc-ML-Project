import tensorflow as tf
import numpy as np

def load_cifar10(batch_size=128, augment=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Standard CIFAR-10 mean/std
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test, 10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    if augment:
        def augment_fn(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, size=[32, 32, 3])
            return image, label
        train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        train_ds
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds