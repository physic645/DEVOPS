# Train the model with optimal hyperparamters


def train_with_optimal_hyperparameters(tuner1,best_hps,best_epoch,X,y,X_test,y_test):
    
    # Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner1.hypermodel.build(best_hps)
    
    # Retrain the model
    hypermodel.fit(X,y, epochs=best_epoch, validation_split=0.2)
    
    
    # Evaluate the hypermodel on the test data.
    
    eval_result = hypermodel.evaluate(X_test, y_test)
    print("[test loss, test accuracy]:", eval_result)
    
    return eval_result[0], eval_result[1]


