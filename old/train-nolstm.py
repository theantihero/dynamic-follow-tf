import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU
import numpy as np
import random
from normalizer_old_seq import norm
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.inf)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf")

with open("data/gm-only/x_train", "r") as f:
    x_train = json.load(f)

with open("data/gm-only/y_train", "r") as f:
    y_train = json.load(f)

NORM = True
if NORM:
    v_ego, v_ego_scale = (norm([i[0] for i in x_train]))
    a_ego, a_ego_scale = (norm([i[1] for i in x_train]))
    v_lead, v_lead_scale = (norm([i[2] for i in x_train])) # not v_lead, v_rel actually
    x_lead, x_lead_scale = (norm([i[3] for i in x_train]))
    a_lead, a_lead_scale = (norm([i[4] for i in x_train]))
    
    x_train = []
    for idx, i in enumerate(v_ego):
        #x_train.append([v_ego[idx], a_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]])
        x_train.append([v_ego[idx], v_lead[idx], x_lead[idx], a_lead[idx]])
    x_train = np.asarray(x_train)
    y_train = np.asarray([np.interp(i, [-1, 1], [0, 1]) for i in y_train])
else:
    x_train = [[i[0], i[2], i[3]] for i in x_train]
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

'''for idx,i in enumerate(y_train):
    if i < -.5 and x_train[idx][0] > 8.9:
        print(i)
        print(idx)
        print(x_train[idx])
        break'''

#opt = keras.optimizers.Adam(lr=0.01, decay=1e-6)
opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.RMSprop(0.001)

'''model = Sequential([
    Dense(5, activation="tanh", input_shape=(x_train.shape[1:])),
    Dense(8, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(32, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(128, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(64, activation="tanh"),
    Dense(32, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(8, activation="tanh"),
    Dense(1),
  ])'''

layer_num=8
nodes=64

model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(x_train.shape[1:])))
for i in range(layer_num):
    model.add(Dense(nodes, activation="relu"))
    #model.add(Dropout(0.08))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_squared_error'])
#tensorboard = TensorBoard(log_dir="logs/{}-custom-{}-custom".format(layer_num, nodes))
model.fit(x_train, y_train, shuffle=True, batch_size=32, epochs=4)  #, callbacks=[tensorboard])

#data = [norm(23.74811363, v_ego_scale), norm(-0.26912481, a_ego_scale), norm(15.10309029, v_lead_scale), norm(55.72000122, x_lead_scale), norm(-0.31268027, a_lead_scale)] #should be -0.5
#prediction=model.predict(np.asarray([[norm(23.74811363, v_ego_scale), norm(15.10309029, v_lead_scale), norm(30.72000122, x_lead_scale)]]))[0][0]
#print((prediction - 0.5)*2.0)

#accur = list([list(i) for i in x_train])

y = [model.predict(np.asarray([[norm(17.8816, v_ego_scale), norm(0, v_lead_scale), norm(i, x_lead_scale), norm(0, a_lead_scale)]]))[0][0] for i in range(100)]
x = [i for i in range(100)]
plt.plot(x,y)
plt.show()

'''accuracy=[]
for i in range(500):
    choice = random.randint(0, len(x_train) - 1)
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
print()
print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(3)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5'''

#test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

#print(model.predict(np.asarray(test_data)))

save_model = True
tf_lite = False
if save_model:
    model_name = "gm-only"
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)