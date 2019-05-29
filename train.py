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

with open("data/all/x_train", "r") as f:
    x_train = json.load(f)

with open("data/all/y_train", "r") as f:
    y_train = json.load(f)

NORM = True
if NORM:
    v_ego, v_ego_scale = (norm([i[0] for i in x_train]))
    a_ego, a_ego_scale = (norm([i[1] for i in x_train]))
    v_lead, v_lead_scale = (norm([i[2] for i in x_train]))
    x_lead, x_lead_scale = (norm([i[3] for i in x_train]))
    a_lead, a_lead_scale = (norm([i[4] for i in x_train]))
    
    x_train = []
    for idx, i in enumerate(v_ego):
        x_train.append([v_ego[idx], a_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]])
    x_train = np.asarray(x_train)
    y_train = np.asarray([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
else:
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

opt = keras.optimizers.Adam(lr=0.001, decay=1e-5)
#opt = keras.optimizers.RMSprop(0.001)

model = Sequential([
    Dense(5, activation="tanh", input_shape=(x_train.shape[1:])),
    Dense(8, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(32, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(75, activation="tanh"),
    Dense(80, activation="tanh"),
    Dense(90, activation="tanh"),
    Dense(100, activation="tanh"),
    Dense(100, activation="relu"),
    Dense(90, activation="relu"),
    Dense(80, activation="relu"),
    Dense(75, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1),
  ])

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.fit(x_train, y_train, shuffle=True, batch_size=16, validation_split=.01, epochs=5)

#accur = list([list(i) for i in x_train])

accuracy=[]
for i in range(500):
    choice = random.randint(0, len(x_train - 1))
    real=y_train[choice]
    to_pred = list(list(x_train)[choice])
    pred = model.predict(np.asarray([to_pred]))[0][0]
    accuracy.append(abs(real-pred))
    #print("Real: "+str(real))
    #print("Prediction: "+str(pred))
    #print()
    
avg = sum(accuracy) / len(accuracy)
if NORM:
    print("Accuracy: "+ str(abs(avg-1)))
else:
    print("Accuracy: "+ str(np.interp(avg, [0, 1], [1, 0])))

#test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

#print(model.predict(np.asarray(test_data)))

save_model = False
tf_lite = False
if save_model:
    model_name = "all-test"
    model.save("models/h5_models/"+model_name+".h5")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)