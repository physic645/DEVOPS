# Multi-class classification with Keras with k-fold cross validation
# Base on BrownLee 

import pandas
from keras.models                import Sequential
from keras.layers                import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils                 import np_utils
from sklearn.model_selection     import cross_val_score
from sklearn.model_selection     import KFold
from sklearn.model_selection     import RepeatedKFold
from sklearn.preprocessing       import LabelEncoder
from sklearn.pipeline            import Pipeline
from sklearn.utils               import shuffle


import numpy             as np
import tensorflow        as tf
import tensorflow_addons as tfa

import random

import keras
import keras.optimizers

from keras.models     import Model
from keras.layers     import Concatenate, Dense, LSTM, Input, concatenate, Layer
#from keras.optimizers import Adagrad
from tensorflow       import keras
from tensorflow.keras import layers
from keras.callbacks  import LambdaCallback

from keras.utils.vis_utils import plot_model # for visualization the model

# Hermitte polynomials

def h0(x):
    return 0*x + 1

def h1(x):
    return 2*x

def h2(x):
    return 4*(x**2) -2

def h3(x):
    return 8*(x**3) - 12*x

def h4(x):
    return 16*(x**4) - 48*(x**2) + 12


class customDense(Layer):
    
    def __init__(self, units, reference_layer, activation = None):  
        super(customDense,self).__init__()
        self.units      = units
        self.ref_layer  = reference_layer
        self.activation = tf.keras.activations.get(activation)
    
    
    def build(self, input_shape):                
        
        # Initialize weights 
        
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name          = "kernel",
                             initial_value = w_init( shape=(input_shape[-1], self.units),dtype = 'float32' ),
                             #initial_value = self.ref_layer.get_weights()[0],
                             trainable     = True )
        
        self.w = self.ref_layer.get_weights()[0] 
        
        # Biases
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name          = "bias",
                             initial_value = b_init(shape=(self.units,), dtype='float32'),
                             trainable     = False)
                             
                             
    def call(self, inputs):
        #self.w = self.ref_layer.get_weights()[0]
        #pass the computation to the activation function
        return self.activation(tf.matmul(inputs, self.w) + self.b)

    
def addPolNodes(numberofnodes,inputs):
    
    numberofstartnodes = int(numberofnodes/3) 
    
    #Initialize
    pol_start = [0 for c in range(numberofstartnodes)]

    #create a 2d array 16x3 filled with zeros
    pol       = [[0 for c in range(3)] for r in range(numberofstartnodes)]
    
    for i in range(numberofstartnodes):
        
        pol_start[i] = Dense(1,activation = h0)
                
        pol[i][0]    = pol_start[i](inputs)
        
        pol[i][1]    = customDense(1,pol_start[i],activation = h1)(inputs)
        
        pol[i][2]    = customDense(1,pol_start[i],activation = h2)(inputs)
        
  
        if i == 0:
            out = concatenate([ pol[i][0],pol[i][1],pol[i][2] ])
        if i > 0:
            out = concatenate([ pol[i][0],pol[i][1],pol[i][2], out ])
            
    
    return out

# load dataset


dataframe = pandas.read_csv("abalone.csv", header=None)


dataset     = dataframe.values

input_shape = dataset.shape[1] - 1

# Delete the first row of the dataset - maybe are labels

dataset = np.delete(dataset,0,0)

# Shuffle the dataset 

#np.random.shuffle(dataset)


X = dataset[1:,0:input_shape].astype(float)
Y = dataset[1:,input_shape]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)



# define baseline model
def baseline_model():
    # create model
    
    
    nn1 = 12
    nn2 = 12 
    nn3 = 12 
    nn4 = 12
    nn5 = 12
    pnn1 = 12
    pnn2 = 9
    
    #Input nodes
    i1  = Input(shape=(input_shape,)) 
          
    d1  = Dense(nn1,activation = 'relu')(i1)
    d2  = Dense(nn2,activation = 'relu')(d1)
    d3  = Dense(nn3,activation = 'relu')(d2)
    d4  = Dense(nn4,activation  = 'relu')(d3)
    d5  = Dense(nn4,activation  = 'relu')(d4)
    
        
    layer1 = addPolNodes(pnn1,d4)
    
    layer2 = addPolNodes(pnn2,layer1)
    
    #layer3 = addPolNodes(pnn,layer2)
    
    #layer4 = addPolNodes(pnn,layer3)
    
   # out   = Dense(dummy_y.shape[1],activation = 'softmax')(layer1)
    
    out   = Dense(dummy_y.shape[1],activation = 'softmax')(d5)
    
    model = Model(inputs = i1, outputs = out)
    
    # Compile model
    #opt =keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9, beta_2=0.999)
    #opt =keras.optimizers.Adamax(learning_rate=500,beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
    return model


n_of_epochs  = 500
b_size       = 10
k_splits     = 5
n_repeats    = 10
random_state = 1
verbose      = 1 


estimator = KerasClassifier(build_fn = baseline_model, epochs = n_of_epochs, batch_size = b_size, verbose = verbose)

# kfold     = KFold(n_splits = k_splits, shuffle=True)

kfold   = RepeatedKFold(n_splits = k_splits, n_repeats = n_repeats, random_state = random_state)

results   = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



# When we want to write the results in a file

filename = 'Typical_DNN.txt' 

f = open(filename, "w")
f.write("Repeated k-fold validation results \n")
f.write("Mean accuracy values \nEach row represents the i iteration and each column the k-folds \n")

for i in range(k_splits*n_repeats):
    results_i = format(results[i],".4f")
    
    
    f.write(f"{ results_i }\t")
    if( (i+1) % k_splits == 0):
        f.write("\n")

f.close()
