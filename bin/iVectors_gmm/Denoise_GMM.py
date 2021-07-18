from keras.layers import BatchNormalization,Dropout,Dense,Input,LeakyReLU
from keras import backend as K
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Model
from keras.utils import plot_model
from keras.initializers import he_normal
from keras.models import model_from_json
from keras import optimizers

def TRAIN_DENOISE(X):
    n_input_dim = X.shape[0]
    # n_output_dim = y_train.shape[1]

    n_hidden1 = 2049
    n_hidden2 = 500
    n_hidden3 = 180

    InputLayer1 = Input(shape=(n_input_dim,), name="InputLayer")
    InputLayer2 = BatchNormalization(axis=1, momentum=0.6)(InputLayer1)

    HiddenLayer1_1 = Dense(n_hidden1, name="H1", activation='relu', kernel_initializer=he_normal(seed=27))(InputLayer2)
    HiddenLayer1_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer1_1)
    HiddenLayer1_3 = Dropout(0.1)(HiddenLayer1_2)

    HiddenLayer2_1 = Dense(n_hidden2, name="H2", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer1_3)
    HiddenLayer2_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer2_1)

    HiddenLayer3_1 = Dense(n_hidden3, name="H3", activation='relu', kernel_initializer=he_normal(seed=65))(HiddenLayer2_2)
    HiddenLayer3_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer3_1)

    HiddenLayer2__1 = Dense(n_hidden2, name="H2_R", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer3_2)
    HiddenLayer2__2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer2__1)

    HiddenLayer1__1 = Dense(n_hidden1, name="H1_R", activation='relu', kernel_initializer=he_normal(seed=27))(HiddenLayer2__2)
    HiddenLayer1__2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer1__1)
    HiddenLayer1__3 = Dropout(0.1)(HiddenLayer1__2)

    OutputLayer = Dense(n_input_dim, name="OutputLayer", kernel_initializer=he_normal(seed=62))(HiddenLayer1__3)

    model = Model(inputs=[InputLayer1], outputs=[OutputLayer])
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0001, amsgrad=False)
    model.compile(loss='mse', optimizer=opt)

    # Y = OutputLayer.reshape(X.shape)
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()

    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
    # fit the model


    hist = model.fit(X, batch_size=512, epochs=100, verbose=1)
    Y = model.predict(X)
    return Y
