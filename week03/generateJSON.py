'''
@Author: Xu Bai
@Date: 2020-07-24 21:35:46
@LastEditors: Xu Bai
@LastEditTime: 2020-07-24 22:32:22
'''
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras.models import model_from_json

print(tf.__version__)
saved_model_path = "saved_models/{}".format(int(time.time()))
model = keras.Sequential([
    keras.layers.Dense(units=1,input_shape=[1])
])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
model.fit(xs,ys,epochs=500)
json_string = model.to_json()
print(json_string)
with open(saved_model_path + ".json",'w') as f:
    f.write(json_string)
print(model.predict([10.0]))