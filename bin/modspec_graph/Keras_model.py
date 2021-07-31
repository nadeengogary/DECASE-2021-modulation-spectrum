import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation, Multiply
from keras import backend as K
from keras import optimizers
########################################################################
# keras model
########################################################################
def get_model(X,X_test):
    n,m,s = X.shape
    X = X.reshape(len(X),-1)
    inputDim = X.shape[1]
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
    # opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0001, amsgrad=False)
    # model.compile(loss='mse', optimizer=opt)
    # model.fit(X)
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0001, amsgrad=False)
    model.compile(loss='mse', optimizer=opt)
    model.fit(X,X,shuffle=True,validation_data=(X_test, X_test))
    Y = model.predict(X_test)
    return Y

def load_model(file_path):
    return keras.models.load_model(file_path)
