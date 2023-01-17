# Train the best model with optimal hyperparamters

# Train the second best that not contains the polynomial

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

def train_with_optimal_hyperparameters(tuner1,best_hps,best_epoch,X,y,X_test,y_test):
    
    # Initiate connection with Neptune AI for monitoring
    run = neptune.init_run(
            project   ='k15redd22/MLOps',
            api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==",
        )
    

    neptune_cbk = NeptuneCallback(run=run)
    
    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner1.hypermodel.build(best_hps)
    
    # Retrain the model
    hypermodel.fit(X,y, epochs=best_epoch, verbose = 0, 
                   validation_data=(X_test, y_test),
                   callbacks=[neptune_cbk])
    
    
    # Evaluate the hypermodel on the test data.
    
    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[Test loss, Test accuracy]:", eval_result)
    
    #run['Test_accuracy'].log(eval_result[1])
    
    run.stop()
    
    return eval_result[0], eval_result[1]


