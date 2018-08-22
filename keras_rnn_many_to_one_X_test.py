#    https://keras.io/getting-started/sequential-model-guide/#examples
#  check this out
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs




from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
import numpy as np

# x_train = np.zeros((1000, None, 2))
# x_train[1, ] = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7]]) 


import numpy as np
from tensorflow.python.keras.models import load_model
model = load_model('/home/young/Tensorflow_projects/coding/keras_rnn_many_to_one_X.h5')
#the batch size of 1 test sample
# x_test_up = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     1.1], 
# 	[0.6,     1.3], [0.7, 1.5], [0.8,     1.7], [0.9,     1.9]]) 
x_test_down_short = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  
x_test_up_down  = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     .6], [0.6,     .5], [0.7, .4], [0.8,     .3], [0.9,     .2]]) 
x_test_up_flat = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     0.9], [0.6,     0.9], [0.7, 0.9], [0.8,     0.9], [0.9,     0.9]]) 
x_test_down_up  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .6], [0.6,  .7], [0.7,  .8], [0.8,  .9], [0.9,  1]]) 
x_test_decreasing  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .5], [0.6,  .4], [0.7,  .3], [0.8,  .2], [0.9,  .1]]) 
x_test_up_short  = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.0, 0.0],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]) 

x_test = np.stack((x_test_down_short, x_test_up_down, x_test_up_flat, x_test_down_up, x_test_decreasing, x_test_up_short))#, axis=1)
y_test = np.array( [2,0,0,1,1, 2] )
#print(x_test)
#print(x_test.shape)


#x_test_up = x_test_up.reshape(1,4,2)
#print( model.predict(x_test_up, batch_size=None, verbose=1) )
scores = model.evaluate(x_test, to_categorical(y_test), verbose=1)
print(scores)
print('training data results: ')
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " +  str(scores[i]*100))


pre_out = model.predict(x_test, batch_size=None, verbose=1)
print(pre_out)
# cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

#post analysis of model fit and evaluate
#https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/


#model.add((Dense(3, activation='sigmoid')))   results:

# 5/5 [==============================]5/5 [==============================] - 0s 27ms/step

# [0.3601095378398895, 0.07460707426071167, 0.800000011920929]
# training data results: 
# loss: 36.01095378398895
# mean_squared_error: 7.460707426071167
# acc: 80.0000011920929

# 5/5 [==============================]5/5 [==============================] - 0s 26ms/step

# [[0.08156254 0.03793901 0.88387424]
#  [0.21539919 0.01396057 0.47214362]
#  [0.73329604 0.02561312 0.06570691]
#  [0.14445683 0.75667053 0.02815693]
#  [0.07201358 0.6943321  0.05677234]]
# [Finished in 4.6s]

# model.add((Dense(3, activation='softmax')))  

# 5/5 [==============================]5/5 [==============================] - 0s 26ms/step

# [0.2547874450683594, 0.04909709841012955, 0.800000011920929]
# training data results: 
# loss: 25.478744506835938
# mean_squared_error: 4.909709841012955
# acc: 80.0000011920929

# 5/5 [==============================]5/5 [==============================] - 0s 27ms/step

# [[6.6355389e-04 2.3135196e-03 9.9702293e-01]
#  [3.0156642e-01 3.2326201e-01 3.7517160e-01]
#  [9.5761818e-01 2.7063403e-02 1.5318362e-02]
#  [7.4732224e-03 9.8555619e-01 6.9705443e-03]
#  [7.2073410e-03 9.8576677e-01 7.0258216e-03]]
# [Finished in 4.6s]
