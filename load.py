import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("C:/Git/dynamic-follow-tf/models/h5_models")
model = tf.keras.models.load_model("15-min-test.h5")
x=[]
y=[]
for i in range(30):
    x.append(i)
    y.append(model.predict(np.asarray([[1,.5,.32,.1,.9]]))[0][0])

plt.plot(x,y)
plt.show()
prediction=model.predict(np.asarray([[1,.5,.32,.1,.9]]))[0][0]
print(prediction)
