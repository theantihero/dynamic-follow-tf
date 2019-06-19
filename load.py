import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from normalizer import norm
os.chdir("C:/Git/dynamic-follow-tf/models/h5_models")
model = tf.keras.models.load_model("gm-only.h5")
v_ego_scale = [0.0, 36.1995735168457]
a_ego_scale = [-3.0412862300872803, 2.78971791267395]
v_lead_scale = [0.0, 91.02222442626953]
x_lead_scale = [0.9600000381469727, 138.67999267578125]
a_lead_scale = [-3.909122943878174, 25.991727828979492]
data = [norm(0, v_ego_scale), norm(0, a_ego_scale), norm(1, v_lead_scale), norm(3, x_lead_scale), norm(.3, a_lead_scale)]
prediction=model.predict(np.asarray([data]))[0][0]
print((prediction - 0.5)*2.0)
