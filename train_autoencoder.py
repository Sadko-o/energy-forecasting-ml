# 2b_train_autoencoder.py
import numpy as np, tensorflow as tf
from pathlib import Path
ART = Path("artifacts")

D = np.load(ART/"reg_data.npz", allow_pickle=True)
Xtr, Xva, Xte = D["Xtr"], D["Xva"], D["Xte"]

tf.random.set_seed(7)
inp = tf.keras.Input(shape=Xtr.shape[1:], name="seq")
x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(inp)
x = tf.keras.layers.MaxPool1D(2)(x)
x = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.UpSampling1D(2)(x)
out = tf.keras.layers.Conv1D(Xtr.shape[-1], 3, padding="same", name="recon")(x)

ae = tf.keras.Model(inp, out)
ae.compile(optimizer="adam", loss="mae")
ae.fit(Xtr, Xtr, validation_data=(Xva, Xva), epochs=40, batch_size=256, verbose=2,
       callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
ae.save(ART/"ae.keras")

# choose threshold on train recon error (e.g., 99th percentile)
recon_tr = ae.predict(Xtr, batch_size=512, verbose=0)
err_tr = np.mean(np.abs(recon_tr - Xtr), axis=(1,2))
thr = float(np.percentile(err_tr, 99))
(Path(ART/"ae_threshold.txt")).write_text(str(thr))
print("Anomaly threshold (MAE):", thr)
