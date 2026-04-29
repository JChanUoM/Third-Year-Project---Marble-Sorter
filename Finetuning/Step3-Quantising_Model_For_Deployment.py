import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("dataset/dataset_for_model/MarbleV3.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

#converter.optimizations = [tf.lite.Optimize.DEFAULT] # will be very slow on windows, but fast in edge device
tflite_model =converter.convert()

with open("dataset/dataset_for_model/MarbleV3.tflite", 'wb') as f:
    f.write(tflite_model)


print("Done")