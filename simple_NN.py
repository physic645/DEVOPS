# A simple NN using tensorflow
'''
import keras
import keras.optimizers
from keras.models                import Sequential
from keras.layers                import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils                 import np_utils
from keras.models                import Model
from keras.layers                import Concatenate, Dense, LSTM, Input, concatenate, Layer
#from keras.optimizers import Adagrad
from tensorflow                  import keras
from tensorflow.keras            import layers
from keras.callbacks             import LambdaCallback
from keras.utils.vis_utils       import plot_model # for visualization the model

from sklearn.model_selection     import cross_val_score
'''

# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=15, batch_size=10)
# evaluate the keras model
# The evaluate() function will return a list with two values. 
# The first will be the loss of the model on the dataset, and the second will be the accuracy of the model on the dataset
loss, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

f = open("demo_results.txt", "w")
f.write('Accuracy: %.2f\n' % (accuracy*100))
f.write('Loss: %.2f\n' % (loss*100))
f.close()



