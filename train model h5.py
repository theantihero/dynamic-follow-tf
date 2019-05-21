import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation
import numpy as np
import random
from normalizer import norm
#np.set_printoptions(threshold=np.inf)

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf")

with open("data/15-min/x_train", "r") as f:
    x_train = json.load(f)

v_ego, v_ego_scale = (norm([i[0] for i in x_train]))
a_ego, a_ego_scale = (norm([i[1] for i in x_train]))
v_lead, v_lead_scale = (norm([i[2] for i in x_train]))
x_lead, x_lead_scale = (norm([i[3] for i in x_train]))
a_lead, a_lead_scale = (norm([i[4] for i in x_train]))

x_train = []
for idx, i in enumerate(v_ego):
    x_train.append([v_ego[idx], a_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]])
x_train = np.asarray(x_train)

with open("data/15-min/y_train", "r") as f:
    y_train = np.asarray(json.load(f))

opt = keras.optimizers.Adam(lr=0.001, decay=1e-5)
#opt = tf.keras.optimizers.RMSprop(0.001)

model = Sequential([
    Dense(32, activation=tf.nn.relu, input_shape=(x_train.shape[1:])),
    Dense(64, activation=tf.nn.relu),
    Dense(128, activation=tf.nn.relu),
    Dense(64, activation=tf.nn.relu),
    Dense(32, activation=tf.nn.relu),
    Dense(8, activation=tf.nn.relu),
    Dense(1),
  ])

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.fit(x_train, y_train, batch_size=1000, epochs=50)

test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

print(model.predict(np.asarray(test_data)))

save_model = True
tf_lite = False
if save_model:
    model_name = "15-min"
    model.save("models/"+model_name+".h5")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/"+model_name+".tflite", "wb").write(tflite_model)