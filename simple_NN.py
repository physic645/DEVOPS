# A simple NN using tensorflow

# Libraries

from p_kaggle import p_kaggle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
 
def simple_NN(nodes_first_layer,epochs,X,y,input_shape):
                    
    run = neptune.init(
        project   ='k15redd22/MLOps',
        api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==",
    )
    

    # number of nodes of first layer    
    nodes_first_layer = nodes_first_layer
            
    # Define the keras model
    model = Sequential()
    model.add(Dense(nodes_first_layer, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    # Use Neptune's Keras callback to log the metrics and artifacts
    neptune_callback = NeptuneCallback(run=run)
    
    
    # fit the keras model on the dataset
    model.fit(X, y, epochs=epochs, batch_size=10,callbacks=[neptune_callback])
    
    # evaluate the keras model
    # The evaluate() function will return a list with two values. 
    # The first will be the loss of the model on the dataset, 
    # and the second will be the accuracy of the model on the dataset
    loss, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    
    return (loss,accuracy)
    
    




