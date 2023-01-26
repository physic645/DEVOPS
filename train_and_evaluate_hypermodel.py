# Train the best model with optimal hyperparamters

# Train the second best that not contains the polynomial

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from keras.callbacks import EarlyStopping

def train_and_evaluate_hypermodel(tuner1,best_hps,epochs,X,y,X_test,y_test):
    
    
    # Initiate connection with Neptune AI for monitoring
    run = neptune.init_run(project ='k15redd22/MLOps', api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==")
    

    neptune_cbk = NeptuneCallback(run=run)
    
    # Build the hypermodel 
    hypermodel = tuner1.hypermodel.build(best_hps)
    
    # Early stopping is a technique used to prevent overfitting by stopping the training process 
    # before the model's performance on the validation dataset stops improving.
    
    # The training process will be stopped if the validation accuracy does not improve for 3 consecutive epochs.
    
    #early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
    
    # Train the model 
    # evaluate the performance on the validation dataset during the training process 
    # (after each training iteration (epoch))
    
    hypermodel.fit(X,y,
                   epochs          = epochs,
                   verbose         = 0,
                   validation_data = (X_test, y_test),
                   callbacks       = [neptune_cbk],#,early_stopping]                  
                   )
    
    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[Test loss, Test accuracy]:", eval_result)
    
    '''
    for i in range(epochs):
    # Evaluate the hypermodel on the test data after the training complete
        eval_result = hypermodel.evaluate(X_test, y_test, 
                                          verbose    = 0, 
                                          batch_size = 32,
                                          callbacks  = [neptune_cbk]
                                         )
    '''
    
    # print("[Test loss, Test accuracy]:", eval_result)
    
    #run['Test_accuracy'].log(eval_result[1])
    
    run.stop()
    
    return 1,2#eval_result[0], eval_result[1]


