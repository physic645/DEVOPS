# Automatic Neural Network Hyperparameter Tuning for TensorFlow Models using Keras Tuner in Python

import tensorflow  as tf
import keras_tuner as kt
import pandas      as pd
#import neptunecontrib.monitoring.kerastuner as npt_utils
import neptune.new as neptune


# Libraries useful to create custom activation function

from keras import backend as K
from keras.layers.core import Activation
#from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import get_custom_objects

def find_optimal_hyperparameters(X,y,input_shape):

    ### Note! You cannot use random python functions, 
    #         activation function gets as an input tensorflow tensors and 
    #         should return tensors. 
    #         There are a lot of helper functions in keras backend.
    def pol_2(x):
      
        return 4*(x**2) -2
        #return 16*(x**4) - 48*(x**2) + 12
         
    get_custom_objects().update({'h_2nd': Activation(pol_2)})
    
        
    #run_nep_ai = neptune.init(
    #        project   ='k15redd22/MLOps',
    #        api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==",
    #    )
    
        
    
    # Build a model that we want to find optimal hyperparameters
    def model_builder(hp):
        
      model = tf.keras.Sequential()
      
      model.add(tf.keras.layers.Flatten(input_shape=(input_shape,)))
    
      
    #  hp_activation    = hp.Choice('activation', values=['relu', 'tanh'])
    #  
    #  hp_layer_1       = hp.Int('layer_1', min_value=1, max_value=100, step=10)
    #  hp_layer_2       = hp.Int('layer_2', min_value=1, max_value=100, step=10)
    #  hp_layer_3       = hp.Int('layer_3', min_value=1, max_value=100, step=10)
    #  hp_layer_4       = hp.Int('layer_4', min_value=1, max_value=100, step=10)
      
      
      hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
      # Try different numbers of layers between 2 and 6
      
      for i in range(hp.Int('layers',2,5)):
          model.add(tf.keras.layers.Dense(
              units      = hp.Int('units_'  + str(i), 50, 100, step=10),
              activation = hp.Choice('act_' + str(i), ['relu', 'tanh', 'h_2nd'])))
                        
    
      
    #  model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation))
    #  model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))
    #  model.add(tf.keras.layers.Dense(units=hp_layer_3, activation=hp_activation))
    #  model.add(tf.keras.layers.Dense(units=hp_layer_4, activation=hp_activation))
      
      
      model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
      model.compile(
          optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
          loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
          metrics   = ['accuracy']
          )
      
      return model
    
    
    '''
    # Using hyperband method
    tuner = kt.Hyperband(
              model_builder,
              objective    = 'val_accuracy',
              max_epochs   = 10,
              factor       = 3,
              #directory    ='dir',
              #project_name ='x',
              #logger = npt_utils.NeptuneLogger()
              )
    '''
    
    
    # Using Bayesian optiization
    tuner1 = kt.BayesianOptimization(
               model_builder,
               objective          = 'val_accuracy',
               max_trials         = 10, # --> max candidates to test
               num_initial_points = 2,
               alpha              = 0.001,
               beta               = 2.6,
               seed               = None,
               overwrite          = True, # --> Overwrite the save data in the below dir
               directory          = 'dir',
               project_name       = 'Bayesian_Optimization',
               )
    
    
    '''
    # Using Random Search method           
    tuner2 = kt.RandomSearch(
        model_builder,
        objective            = 'val_accuracy',
        max_trials           = 5,
        executions_per_trial = 1,
        #directory            = 'dir',
        #project_name         = 'random_search',
    )
    '''
    
    
    
    # Print the tuner search space 
    # tuner2.search_space_summary()
    
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    tuner1.search(X, y, epochs=5, validation_split=0.2, callbacks=[stop_early]) #validation_data=(x_test, y_test))
    
    
    # Get the optimal hyperparameters 
    best_hps = tuner1.get_best_hyperparameters(num_trials=5)[0]
    
    
    
    '''
    print(f"""
    The hyperparameter search has been complete.\n
    The optimal hyperparamnumber are:\n
    # of units for the first  densely-connected layer is {best_hps.get('layer_1')}\n 
    # of units for the second densely-connected layer is {best_hps.get('layer_2')}\n
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}\n
    The optimal actvation function is {best_hps.get('activation')}
    """)
    '''
    
        
    # Find the optimal number of epochs to train the model 
    # with the hyperparameters obtained from the search.
    model   = tuner1.hypermodel.build(best_hps)
    
    history = model.fit(X, y, epochs=50, validation_split=0.2,verbose=2)#,callbacks=[stop_early])
    
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch        = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    #print('Best epoch: %d' % (best_epoch,))
    
    return tuner1,best_hps,best_epoch    