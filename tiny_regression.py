# 2_train_regression.py
import numpy as np, tensorflow as tf, json
from pathlib import Path

ART = Path("artifacts")

data = np.load(ART/"reg_data.npz", allow_pickle=True)
Xtr, ytr = data["Xtr"], data["ytr"]
Xva, yva = data["Xva"], data["yva"]
Xte, yte = data["Xte"], data["yte"]

tf.random.set_seed(7)

inp = tf.keras.Input(shape=Xtr.shape[1:], name="seq")  # (HIST, n_feats)

# --- tiny depthwise CNN encoder ---
x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=3,
        padding="same"          # <- changed from "causal"
    )(inp)
x = tf.keras.layers.Conv1D(
        16,                     # pointwise 1x1 conv to mix channels
        kernel_size=1,
        activation="relu"
    )(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)
out = tf.keras.layers.Dense(1, name="y")(x)

model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mae",
    metrics=["mse"],
)

model.summary()

cb = [
    tf.keras.callbacks.EarlyStopping(
        patience=8,
        restore_best_weights=True,
        monitor="val_loss"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=4,
        factor=0.5
    )
]

model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=50,
    batch_size=256,
    callbacks=cb,
    verbose=2
)

print("Test MAE/MSE:", *model.evaluate(Xte, yte, verbose=0))
model.save(ART/"reg_keras.keras")
