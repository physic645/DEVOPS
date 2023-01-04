# python runner script

# Libraries
from connect_with_kaggle                import connect_with_kaggle
from find_optimal_hyperparameters       import find_optimal_hyperparameters
from train_with_optimal_hyperparameters import train_with_optimal_hyperparameters
from scipy                              import stats
import neptune.new as neptune
import time
from sklearn.model_selection            import train_test_split # split a dataset into train and test sets
start = time.time()

'''
# Initiate connection with Neptune AI for monitoring
run_nep_ai = neptune.init(
        project   ='k15redd22/MLOps',
        api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==",
    )
'''

# Step:1
# Connect_with_kaggle and download the working dataset once

searchname      = "diabetes"
X,y,input_shape = connect_with_kaggle(searchname)

# split into train test sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.20)


# Step:2
# Find the optimal hyperparameter for the speficic dataset

tuner1,best_hps,best_epoch = find_optimal_hyperparameters(X,y,input_shape)

# Step:3
# Train the hypermodel with optimal hyperparamters

train_with_optimal_hyperparameters(tuner1,best_hps,best_epoch,X,y,X_test,y_test)





# Stops the connection to Neptune and synchronizes all data.
# run_nep_ai.stop()

end = time.time()
total = end-start

import connect_with_kaggle

print(f'\n We trained the {connect_with_kaggle.title} dataset made by {connect_with_kaggle.creator} \n')
print(f'This neural network training took {total:.3f} seconds')