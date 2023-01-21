# Automatic Neural Network Hyperparameter Tuning for TensorFlow Models using Keras Tuner in Python

import tensorflow  as tf
import keras_tuner as kt
import pandas      as pd
#import neptunecontrib.monitoring.kerastuner as npt_utils
import neptune.new as neptune
import random


# Libraries useful to create custom activation function

from keras import backend as K
from keras.layers.core import Activation
#from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import get_custom_objects



def find_optimal_hyperparameters(X,y,input_shape,X_test,y_test):

    
    def pol_2(x):
      
        return 4*(x**2) -2
        #return 16*(x**4) - 48*(x**2) + 12
         
    get_custom_objects().update({'h_2nd': Activation(pol_2)})
    
        
    # Build a model that we want to find optimal hyperparameters
    def model_builder(hp):
        
      model = tf.keras.Sequential()      
      model.add(tf.keras.layers.Flatten(input_shape=(input_shape,)))          
      hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
      # Try different numbers of layers between 2 and 6
      
      for i in range(hp.Int('layers',2,5)):
          model.add(tf.keras.layers.Dense(
              units      = hp.Int('units_'  + str(i), 20, 100, step=5),
              activation = hp.Choice('act_' + str(i), ['relu', 'tanh', 'h_2nd'])))
                        
           
      model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
      model.compile(
          optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
          loss      = tf.keras.losses.BinaryCrossentropy(from_logits=False),
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
    
    max_trials = 100
    epochs     = 100
    
    # Using Bayesian optiization
    tuner1 = kt.BayesianOptimization(
               model_builder,
               objective          = 'val_accuracy',
               max_trials         = max_trials, # --> max candidates to test
               num_initial_points = 2,
               alpha              = 0.001,
               beta               = 2.6,
               seed               = random.seed(), # --> makes the entire optimization process different every time we run it.            
               #overwrite          = True, # --> Overwrite the save data in the below dir
               #directory          = 'dir',
               #project_name       = 'Bayesian_Optimization',
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
    
    
    #stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    tuner1.search(X, y, epochs=epochs, verbose=0,validation_data=(X_test, y_test))#,callbacks=[stop_early])
    
    
    # Get the optimal hyperparameters 
    best_hps    = tuner1.get_best_hyperparameters(num_trials = max_trials)[0]
    second_best = tuner1.get_best_hyperparameters(num_trials = max_trials)[1]
    
    
    for i in range(max_trials):
        
        a = tuner1.get_best_hyperparameters(num_trials = max_trials)[i]
            
        # Dictionary
        v = a.values
        
        if 'h_2nd' not in v.values():
            scenario_without_pol = tuner1.get_best_hyperparameters(num_trials = max_trials)[i]
            break
    
    
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
    
    '''
    # Find the optimal number of epochs to train the model 
    # with the hyperparameters obtained from the search.
    model   = tuner1.hypermodel.build(best_hps)
    
    history = model.fit(X, y, epochs=epochs, 
                        #validation_split=0.2,
                        verbose         = 0 ,                        
                        validation_data = (X_test, y_test),
                        #callbacks       = [stop_early],
                        )
    
    val_acc_per_epoch = history.history['val_accuracy']
    
    #run['validation_accuracy_per_epoch'].log(val_acc)
    
    best_epoch        = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    '''
    best_epoch = 1
    #run.stop()
    
    return tuner1,best_hps,best_epoch,second_best,scenario_without_pol
