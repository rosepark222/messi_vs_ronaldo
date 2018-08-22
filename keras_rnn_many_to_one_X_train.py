#    https://keras.io/getting-started/sequential-model-guide/#examples
#  check this out
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
import numpy as np
 
from tensorflow.python.keras.preprocessing import sequence 


model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(None, 2)))
model.add(LSTM(8, return_sequences=False)) 
#model.add((Dense(3, activation='sigmoid')))  
model.add((Dense(3, activation='softmax')))  

#The sigmoid activation is outputing values between 0 and 1 independently from one another.
#If you want probabilities outputs that sums up to 1, use the softmax activation on your last layer, it will normalize the output to sum up to 1. 

#model.add((Dense(2, activation='softmax')))  

	#timedistributed requires all sequences (return_sequences = True). 
	#https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',  metrics=['mse', 'accuracy'])
#    https://github.com/keras-team/keras/issues/2548
#    evaluate() will return the list of metrics that the model was compiled with. 
#    So if your compile metrics list include 'accuracy', then this should still work.


# Our array is:
# [[3 7]
#  [9 1]]

# Applying sort() function: last axis is column
# [[3 7]
#  [1 9]]

# Sort along axis 0, which is row
# [[3 1]
#  [9 7]]

def messi_gen():
   x = np.sort ( np.random.rand(1, 10) )
   slope = .5 * np.random.randn(1) + 1
   intercept = .2 * np.random.randn(1)
   y = slope * x + intercept
   return (np.hstack((x.T,y.T)))

def ronaldo_gen():
   x = np.sort ( np.random.rand(1, 10) )
   slope = -1* (.5 * np.random.randn(1) + 1)
   intercept = .2 * np.random.randn(1) + 1
   y = slope * x + intercept
   return (np.hstack((x.T,y.T)))

mask = np.array([[1,   1], [1,   1],[1,   1],[1,   1],[1,   1],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

def train_generator():
  while True:
#        sequence_length = np.random.randint(10, 100)
        sequence_length = 10 #10 dots per stroke
        x_train = np.zeros((1000, sequence_length, 2)) # x and y points
        y_train = np.random.randint(3, size=1000)
        #expected dense_1 to have shape
        #print( "AAB ")
        for i in range(0, 1000):
           if(y_train[i] == 0):
             x_train[i, ] = messi_gen() #np.array([[0,	0.1], [0.1,	0.3], [0.2,	0.5], [0.3,	0.7], [0.4,	0.9], [0.5,	1.1], [0.6,	1.3], [0.7,	1.5], [0.8,	1.7], [0.9,	1.9]]) 
           elif(y_train[i] == 1):
             x_train[i, ] = ronaldo_gen() #np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .5], [0.6,  .4], [0.7,  .3], [0.8,  .2], [0.9,  .1]]) 
           elif(y_train[i] == 2):
             x_train[i, ] = messi_gen() * mask #np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  
#           else:

           if(i == 500*100):
             print(i)
             print(x_train.shape)
             print(x_train[i,])
             print(y_train[i])

        yield x_train, to_categorical(y_train)

model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)


model.save('/home/young/Tensorflow_projects/coding/keras_rnn_many_to_one_X.h5')


# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# import numpy as np
# from tensorflow.python.keras.models import load_model
# model = load_model('/home/young/Tensorflow_projects/coding/keras_rnn_many_to_one_X.h5')
# #the batch size of 1 test sample
# x_test_up = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     1.1], [0.6,     1.3], [0.7, 1.5], [0.8,     1.7], [0.9,     1.9]]) 
# x_test_up = x_test_up.reshape(1,10,2)
# print( model.predict(x_test_up, batch_size=None, verbose=1) )
# #   array([[0.00663349, 0.6025158 ]], dtype=float32)   -> 0/1 , 1 is greater -> up sloping pattern

# x_test_decreasing  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .5], [0.6,  .4], [0.7,  .3], [0.8,  .2], [0.9,  .1]]) 
# x_test_decreasing= x_test_decreasing.reshape(1,10,2)
# model.predict(x_test_decreasing, batch_size=None, verbose=1)
# #   array([[0.5322375 , 0.00834433]], dtype=float32)   -> 0/1 , 0 is greater -> ----> correctly identifying it is decreasing!!!!

# x_test_down_up  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .6], [0.6,  .7], [0.7,  .8], [0.8,  .9], [0.9,  1]]) 
# x_test_down_up= x_test_down_up.reshape(1,10,2)
# model.predict(x_test_down_up, batch_size=None, verbose=1)
# #   array([[0.5102862 , 0.00811925]], dtype=float32)   -> 0/1 , 0 is greater ->-----> down and then up --- recognized as down

# x_test_up_down  = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     .6], [0.6,     .5], [0.7, .4], [0.8,     .3], [0.9,     .2]]) 
# x_test_up_down= x_test_up_down.reshape(1,10,2)
# model.predict(x_test_up_down, batch_size=None, verbose=1)
# #   array([[0.3975516, 0.011379 ]], dtype=float32)     -> 0/1 , 0 is greater ->---> this also down? 

# x_test_up_flat = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     0.9], [0.6,     0.9], [0.7, 0.9], [0.8,     0.9], [0.9,     0.9]]) 
# x_test_up_flat = x_test_up_flat.reshape(1,10,2)
# model.predict(x_test_up_flat, batch_size=None, verbose=1)
# #   array([[0.0125859 , 0.42021653]], dtype=float32)   -> 0/1 , 1 is greater -> recognized as up sloping
 




