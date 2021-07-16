import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Multiply
from keras import backend as K

########################################################################
# keras model
########################################################################
def get_model(X):
    # inputDim = X.shape[1]
    inputLayer = Input(shape=(inputDim,))

    h = Dense(128)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim)(h)
    model = Model(inputs=inputLayer, outputs=h)
    # h.reshape(X.shape)
    # model.compile()
    # return model.predict(X)
    # return Model(inputs=inputLayer, outputs=h)
    return np.array(h.shape())

def load_model(file_path):
    return keras.models.load_model(file_path)
