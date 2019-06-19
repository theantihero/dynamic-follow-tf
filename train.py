import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, BatchNormalization, LeakyReLU, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Reshape
import numpy as np
import random
from normalizer import norm
from normalizer import get_3d_min_max
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
import time
#np.set_printoptions(threshold=np.inf)

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf")

with open("data/LSTM/x_train-gbergman", "r") as f:
    x_train = json.load(f)

with open("data/LSTM/y_train-gbergman", "r") as f:
    y_train = json.load(f)

NORM = True
if NORM:
    #x_train_copy = list(x_train)
    v_ego_scale = get_3d_min_max(x_train, 0)
    a_ego_scale = get_3d_min_max(x_train, 1)
    v_lead_scale = get_3d_min_max(x_train, 2)
    x_lead_scale = get_3d_min_max(x_train, 3)
    a_lead_scale = get_3d_min_max(x_train, 4)
    
    x_train_reformat = [] # reformat to sequence data for nn to train, to test
    for idx, i in enumerate(x_train):
        x_train_reformat.append([[], [], [], [], []])
        for x in i:
            x_train_reformat[idx][0].append(norm(x[0], v_ego_scale))
            x_train_reformat[idx][1].append(norm(x[1], a_ego_scale))
            x_train_reformat[idx][2].append(norm(x[2], v_lead_scale))
            x_train_reformat[idx][3].append(norm(x[3], x_lead_scale))
            x_train_reformat[idx][4].append(norm(x[4], a_lead_scale))
    
    x_train = x_train_reformat
    
    #x_train = x_train * 2
    #y_train = y_train * 2
    x_train = np.asarray(x_train)
    y_train = np.asarray([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
else:
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

'''for idx,i in enumerate(y_train):
    if i < -.5 and x_train[idx][0] > 8.9:
        print(i)
        print(idx)
        print(x_train[idx])
        break'''

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
#opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.RMSprop(0.001)

'''model = Sequential([
    Dense(5, activation="tanh", input_shape=(x_train.shape[1:])),
    Dense(8, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(300, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(8, activation="tanh"),
    Dense(1),
  ])'''

model = Sequential()
model.add(CuDNNLSTM(40, input_shape=(x_train.shape[1:]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

for i in range(4):
    model.add(CuDNNLSTM(40, return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
model.add(CuDNNLSTM(40))
model.add(Dense(1))


'''model.add(LSTM(64, return_sequences=True, activation="tanh", input_shape=(x_train.shape[1:])))
model.add(Flatten())
for i in range(20):
    model.add(Dense(40, activation="tanh"))
model.add(Dense(1, activation="linear"))'''

model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_squared_error'])
#tensorboard = TensorBoard(log_dir="logs/test-{}".format("30epoch"))
model.fit(x_train, y_train, shuffle=True, batch_size=32, validation_split=.02, epochs=5) # callbacks=[tensorboard]

data = [[0.04548478, 0.04523729, 0.04512114, 0.04488637, 0.04469746,
        0.04454154, 0.0443153 , 0.04414989, 0.04390321, 0.04371238,
        0.04350156, 0.04326949, 0.04317012, 0.0429304 , 0.04273162,
        0.04249655, 0.04220375, 0.04202759, 0.04182816, 0.04158718],
       [0.02413954, 0.02384957, 0.02376735, 0.02354851, 0.02346166,
        0.02331117, 0.02316262, 0.02309356, 0.02292289, 0.02281641,
        0.02275922, 0.02266193, 0.02270319, 0.02259346, 0.02249625,
        0.02232534, 0.02213737, 0.02203436, 0.02199109, 0.0219595 ],
       [0.02658086, 0.02658086, 0.02658086, 0.02658086, 0.02658086,
        0.02728457, 0.02658086, 0.02658086, 0.02658086, 0.02658086,
        0.02658086, 0.02658086, 0.02658086, 0.02658086, 0.02658086,
        0.02658086, 0.02658086, 0.02658086, 0.02658086, 0.02658086],
       [0.10439208, 0.10340868, 0.10270624, 0.10144185, 0.10059893,
        0.09947502, 0.09881941, 0.09821064, 0.09708674, 0.0963843 ,
        0.09451114, 0.0941365 , 0.09329357, 0.09245065, 0.09142041,
        0.09048382, 0.08992187, 0.08917261, 0.08889163, 0.08823603],
       [0.02684641, 0.02684641, 0.02684641, 0.02684641, 0.02684641,
        0.0267554 , 0.02684641, 0.02684641, 0.02684641, 0.02684641,
        0.02684641, 0.02684641, 0.02684641, 0.02684641, 0.02684641,
        0.02684641, 0.02684641, 0.02684641, 0.02684641, 0.02684641]]
prediction=model.predict(np.asarray([data]))[0][0]  # should be 0.5

#print((prediction - 0.5)*2.0) if NORM else print(prediction)
print(prediction)


#accur = list([list(i) for i in x_train])

try:
    accuracy=[]
    for i in range(500):
        choice = random.randint(0, len(x_train - 2))
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
except:
    pass
    


#test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

#print(model.predict(np.asarray(test_data)))

save_model = False
tf_lite = False
if save_model:
    model_name = "LSTM-gbergman"
    model.save("models/h5_models/"+model_name+".h5")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)