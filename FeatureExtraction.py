from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
import numpy as np
from DataProcessing import test_Y, train_Y, test_normX, train_normX
from keras import Sequential
from keras.regularizers import l1


# define encoder
n_inputs = train_normX.shape[1]
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs * 2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
n_bottleneck = round(float(n_inputs) / 3.0)
# n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)

# define decoder, level 1
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(n_inputs * 2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='sigmoid')(d)

# define autoencoder model
model = Model(inputs=visible, outputs=output)

# compile autoencoder model
model.compile(optimizer='adam', loss='mse')

# fit the autoencoder model to reconstruct input
hist = model.fit(train_normX, train_normX, epochs=150, batch_size=16, verbose=2,
                 validation_data=(test_normX, test_normX))

# plot loss
pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='test')
limits = [0, 50, 0, .004]
pyplot.axis(limits)
pyplot.legend()
pyplot.xlabel("Epochs")
pyplot.ylabel("MSE")
pyplot.show()

# Define the encoder model (without decoder)
encoder = Model(inputs=visible, outputs=bottleneck)
# save the encoder to file
encoder.save('encoder.h5')


# # Build PCA style autoencoder
# nb_epoch = 1000
# batch_size = 16
# input_dim = train_normX.shape[1]
# encoding_dim = round(float(input_dim) / 3.0)
# #learning_rate = 0.001
#
# encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias = True)
# decoder = Dense(input_dim, activation="linear", use_bias = True)
#
# autoencoder = Sequential()
# autoencoder.add(encoder)
# autoencoder.add(decoder)
#
# autoencoder.compile(loss='mse',optimizer='sgd')
# autoencoder.summary()
#
# hist = autoencoder.fit(train_normX, train_normX,epochs=nb_epoch, batch_size=batch_size, shuffle=True,
#                     verbose=2, validation_data=(test_normX, test_normX))
#
# # plot loss
# pyplot.plot(hist.history['loss'], label='train')
# pyplot.plot(hist.history['val_loss'], label='test')
# limits = [0, 1000, 0, .004]
# pyplot.axis(limits)
# pyplot.legend()
# pyplot.xlabel("Epochs")
# pyplot.ylabel("MSE")
# pyplot.show()


# # Sparse Stacked Autoencoder
# # define encoder
# n_inputs = train_normX.shape[1]
# visible = Input(shape=(n_inputs,))
# n_bottleneck = round(float(n_inputs) / 3.0)
# encoded = Dense(124, activation='relu',activity_regularizer=l1(10e-7))(visible)
# encoded = Dense(64, activation='relu',activity_regularizer=l1(10e-7))(encoded)
# bottleneck = Dense(n_bottleneck, activation='relu',activity_regularizer=l1(10e-7))(encoded)
#
# decoded = Dense(64, activation='relu')(bottleneck)
# decoded = Dense(128, activation='relu')(decoded)
# output = Dense(n_inputs, activation='sigmoid')(decoded)
#
# # define autoencoder model
# model = Model(inputs=visible, outputs=output)
#
# # compile autoencoder model
# model.compile(optimizer='adam', loss='mse')
#
# # fit the autoencoder model to reconstruct input
# hist = model.fit(train_normX, train_normX, epochs=50, batch_size=16, verbose=2,
#                  validation_data=(test_normX, test_normX))
#
# # plot loss
# pyplot.plot(hist.history['loss'], label='train')
# pyplot.plot(hist.history['val_loss'], label='test')
# limits = [0, 50, 0, .004]
# pyplot.axis(limits)
# pyplot.legend()
# pyplot.xlabel("Epochs")
# pyplot.ylabel("MSE")
# pyplot.show()
