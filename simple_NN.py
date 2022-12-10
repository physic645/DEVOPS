# A simple NN using tensorflow

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
