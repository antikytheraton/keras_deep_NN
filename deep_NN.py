import numpy as np
import keras
import utils

import matplotlib.pyplot as plt

print('Keras : {}'.format(keras.__version__))

# Carga dataset mnist
utils.load_mnist

# One-hot encoding de etiquetas
keras.utils.np_utils.to_categorical

# crea el modelo de keras
keras.models.Sequential

keras.layers.InputLayer
keras.layers.Dense
keras.layers.Activation

# entrena el modelo por Gradiente de descenso
keras.optimizers.SGD

(x_train, y_train), (x_test, y_test) = utils.load_mnist(path='mnist/')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape([-1, 28*28])
x_test = x_test.reshape([-1, 28*28])

# labels
from keras.utils.np_utils import to_categorical
y_train_enc = to_categorical(y_train, num_classes=10)
y_test_enc = to_categorical(y_test, num_classes=10)


# contruccion de la red neuronal
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Activation

# se instancia el modelo
model = Sequential()

model.add(InputLayer(input_shape=(784,)))

model.add(Dense(256, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))

model.add(Dense(10, activation='softmax'))

model.summary()


from keras.optimizers import SGD
sdg = SGD(lr=0.0005, momentum=0.9)

model.compile(optimizer=sdg,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_enc,
                batch_size=32, epochs=200,
                verbose=2,
                validation_split=0.2,
                shuffle=True)


# Registry of training process
plt.plot(history.history['acc'],'r')
plt.plot(history.history['val_acc'],'b')
plt.ylabel('Accuracy %')
plt.xlabel('Training iterations (epochs)')
plt.legend(['Training','Testing'], loc='upper left')

plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'b')
plt.ylabel('Loss %')
plt.xlabel('Training iterations (epochs)')
plt.legend(['Training', 'Testing'], loc='upper left')
plt.show()


# Evaluacion del modelo
res = model.evaluate(x_test, y_test_enc)
print('\n[loss, accuracy] : {}'.format(res))
model.save('models/model_deep_NN_aaron.h5')
