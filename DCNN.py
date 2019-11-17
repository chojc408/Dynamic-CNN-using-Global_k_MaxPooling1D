import numpy as np
np.random.seed(999)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D


class Global_k_MaxPooling1D(Layer):
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[1] * self.k))
    def call(self, inputs):
        inputs = tf.transpose(inputs, [0, 2, 1]) # To be sorted along the sequence (time) dimension
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=True, name=None)[0]
        top_k = tf.transpose(top_k, [0, 2, 1]) # To generate GlbalMaxPooling1D-like tensor  
        return Flatten()(top_k)

def CNN():
    # Conventional CNN for text analysis
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=max_len))
    model.add(Conv1D(num_filters, kernel_size, padding='same', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(500, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.summary()
    return model

def KCNN(k):
    # CNN using Global_k_max_Pooling1D (Preliminary version)
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=max_len))
    model.add(Conv1D(num_filters, kernel_size, padding='same', strides=1))
    model.add(Global_k_MaxPooling1D(k))
    model.add(Dense(500, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.summary()
    return model

def LossPlot():
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'c', linestyle = '-', label='Train Loss')
    loss_ax.plot(hist.history['val_loss'], 'c', linestyle = ':', label='Validation Loss')
    loss_ax.set_ylim([0.0, 3.0])
    acc_ax.plot(hist.history['accuracy'], 'k', linestyle = '-', label='Train Accuracy')
    acc_ax.plot(hist.history['val_accuracy'], 'k', linestyle = ":", label='Validation Accuracy')
    acc_ax.set_ylim([0.0, 1.0])
    loss_ax.set_xlabel('Epoch', fontsize = 12)
    loss_ax.set_ylabel('Loss', fontsize = 12)
    acc_ax.set_ylabel('Accuracy', fontsize = 12)
    loss_ax.legend(loc='lower left')
    acc_ax.legend(loc='upper left')
    plt.show()

# === Hyperparameters ====
num_words     = 1000
max_len       = 1000

embedding_dim = 64
num_filters   = 128  # Number of convolution filters
kernel_size   = 3    # Convolution filter size

# ========= Model Compile ========
model = CNN()
# model = KCNN(k=3) # If k == 1, then KCNN == CNN 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

hist = model.fit(X_train, T_train,
                 epochs = 5,
                 batch_size = 64,
                 validation_data = (X_val, T_val),
                 verbose = 1)
