import os
import tensorflow as tf
from src.model import build_model
from src.data import load_cifar10

def main():
    print("Loading CIFAR-10...")
    train_ds, test_ds = load_cifar10(batch_size=128, augment=True)

    print("Building model...")
    model = build_model(channels=128, depth=6, kernel_size=5)
    print(f"Total parameters: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    def lr_schedule(epoch, lr):
        if epoch < 80:
            return 1e-3
        elif epoch < 120:
            return 1e-4
        else:
            return 1e-5

    os.makedirs("models", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_weights.weights.h5",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=150,
        callbacks=callbacks
    )

    print("\nFinal evaluation on test set:")
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()