import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model("/Git/dynamic-follow-tf/dynamic_follow_model")
prediction=model.predict(np.asarray([[[31., 60., 22]]]))
print(prediction)
print(model.predict(np.asarray([[[0.0, 6.840000152587891, 0.0]]])))