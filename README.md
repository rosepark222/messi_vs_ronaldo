# messi_vs_ronaldo
LTSM RNN traning and testing for the traces for messi (left bottom to right top) and ronaldo (left top to right bottom).

It trains 10 (x,y) trance points to classfy in one of three -- messi, ronaldo, and short messi (masking last 5 points to zeros).

batch size = 1000
batches per epoch = 30
epoch = 10

model.compile(loss='categorical_crossentropy',
                optimizer='adam',  metrics=['mse', 'accuracy'])
            
__________________________________________________________________________________________
Layer (type)                            Output Shape                        Param #       
==========================================================================================
lstm_1 (LSTM)                           (None, None, 32)                    4480          
__________________________________________________________________________________________
lstm_2 (LSTM)                           (None, 8)                           1312          
__________________________________________________________________________________________
dense_1 (Dense)                         (None, 3)                           27            
==========================================================================================
Total params: 5,819
Trainable params: 5,819
Non-trainable params: 0
__________________________________________________________________________________________

Epoch 1/10
30/30 [==============================]30/30 [==============================] - 3s 110ms/step - loss: 1.0449 - mean_squared_error: 0.2110 - acc: 0.4766
Epoch 2/10
30/30 [==============================]30/30 [==============================] - 2s 78ms/step - loss: 0.8952 - mean_squared_error: 0.1800 - acc: 0.6271
Epoch 3/10
30/30 [==============================]30/30 [==============================] - 2s 82ms/step - loss: 0.7418 - mean_squared_error: 0.1473 - acc: 0.6748
Epoch 4/10
30/30 [==============================]30/30 [==============================] - 3s 84ms/step - loss: 0.4784 - mean_squared_error: 0.0904 - acc: 0.8472
Epoch 5/10
30/30 [==============================]30/30 [==============================] - 2s 79ms/step - loss: 0.2147 - mean_squared_error: 0.0283 - acc: 0.9827
Epoch 6/10
30/30 [==============================]30/30 [==============================] - 2s 79ms/step - loss: 0.1178 - mean_squared_error: 0.0102 - acc: 0.9946
Epoch 7/10
30/30 [==============================]30/30 [==============================] - 2s 79ms/step - loss: 0.0711 - mean_squared_error: 0.0040 - acc: 0.9980
Epoch 8/10
30/30 [==============================]30/30 [==============================] - 2s 79ms/step - loss: 0.0489 - mean_squared_error: 0.0020 - acc: 0.9989
Epoch 9/10
30/30 [==============================]30/30 [==============================] - 2s 79ms/step - loss: 0.0388 - mean_squared_error: 0.0014 - acc: 0.9988
Epoch 10/10
30/30 [==============================]30/30 [==============================] - 2s 78ms/step - loss: 0.0317 - mean_squared_error: 9.7086e-04 - acc: 0.999
