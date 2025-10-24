# 3_quantize_int8.py
import numpy as np, tensorflow as tf
from pathlib import Path
ART = Path("artifacts")

Xtr = np.load(ART/"reg_data.npz")["Xtr"].astype(np.float32)
rep = (Xtr[i:i+1] for i in range(0, min(2000, len(Xtr)), 5))  # light sampler

model = tf.keras.models.load_model(ART/"reg_keras.keras")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = lambda: rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()

(ART/"energy_nextmin_int8.tflite").write_bytes(tfl)
print("Wrote", ART/"energy_nextmin_int8.tflite")
