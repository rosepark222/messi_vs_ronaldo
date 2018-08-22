# messi_vs_ronaldo
LTSM RNN traning and testing for the traces for messi (left bottom to right top) and ronaldo (left top to right bottom).

It trains 10 (x,y) trance points to classfy in one of three -- messi, ronaldo, and short messi (masking last 5 points to zeros).

batch size = 1000
batches per epoch = 30
epoch = 10

model.compile(loss='categorical_crossentropy',
                optimizer='adam',  metrics=['mse', 'accuracy'])
            
