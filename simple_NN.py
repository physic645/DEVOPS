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
def simple_NN(nodes_first_layer,epochs):
    
    # Libraries
    from numpy                   import loadtxt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense 
    import kaggle 
    import zipfile
    import numpy as np
    import pandas as pd
    import os
    
    # Replace the manual laoding procedure with one more automated ------------------------------------------
    
    # load the dataset
    # dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
        
    # Second authentication method via os
    # Perform the authentication in the notebook directly by using the OS environment variables
    
    os.environ['KAGGLE_USERNAME'] = "konstantinosfilippou"
    os.environ['KAGGLE_KEY']      = "3514308d4ba9316c4f8b7bd9ecc245fb"

    # Connect and initialize the API
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    # Search a dataset via various criterions (maxsize in bytes)
    datasets = kaggle.api.dataset_list(search="heart",max_size="10000000")
    print(datasets)

    # List all metadata of the first in list info using the vars() function
    ds      = datasets[1]
    ds_vars = vars(ds)
    for var in ds_vars:
        print(f"{var} = {ds_vars[var]}")
        
    # Download the zip file of the dataset in dataset_download folder (is created automatically)
    # ds.ref is the path of the kaggle dataset 'noahgift/social-power-nba'
    api.dataset_download_files(ds.ref,path='./dataset_download')
    
    # Unzip the zip 
    
    # -> the last part of the url has always the same format, so we use it to create the zip name.
    zip_name = ds.url.split('/')[-1] 

    with zipfile.ZipFile('./dataset_download/' + zip_name + '.zip','r') as zipref:
        zipref.extractall('./dataset_download')
        
    
    # List files that exist inside the zip 
    files = kaggle.api.dataset_list_files(ds.ref).files
    print(files)
    
    # Process the csv file
    
    dataframe = pd.read_csv(r'./dataset_download/' + str(files[0]))
    dataset     = dataframe.values

    # Delete the first row of the dataset - maybe are labels - if not we are losing one row 
    dataset = np.delete(dataset,0,0)
    input_shape = dataset.shape[1] - 1
    X = dataset[:,0:input_shape]#.astype(float)
    y = dataset[:,input_shape]
        
    # -----------------------------------  END OF DOWNLOAD AND UNZIP  -----------------------------------------
    
    # number of nodes of first layer
    
    nodes_first_layer = nodes_first_layer
    
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]
    
    # define the keras model
    model = Sequential()
    model.add(Dense(nodes_first_layer, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit the keras model on the dataset
    model.fit(X, y, epochs=epochs, batch_size=10)
    
    # evaluate the keras model
    # The evaluate() function will return a list with two values. 
    # The first will be the loss of the model on the dataset, and the second will be the accuracy of the model on the dataset
    loss, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    
    return (loss,accuracy)
    
    




