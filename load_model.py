import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("C:/Git/dynamic-follow-tf/models")
model = tf.keras.models.load_model("15-min.h5")
x=[]
y=[]
for i in range(30):
    x.append(i)
    y.append(model.predict(np.asarray([[30, 0, 30, 30-i, -.5]]))[0][0])

plt.plot(x,y)
plt.show()
#prediction=model.predict(np.asarray([[31., 60., 60]]))
#print(prediction)
