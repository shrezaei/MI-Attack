#experiment with different setting and save in different epochs
from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import argparse
import h5py
import json
import os
import tensorflow as tf
from keras.models import model_from_config


parser = argparse.ArgumentParser(description='Train a target model.')
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'imagenet_inceptionv3', 'imagenet_xception'],  help='Indicate dataset and target model archtecture.')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('-e', '--epochs', type=int, default=40, help='Number of training epochs')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate.')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    save_dir = os.path.join(os.getcwd(), 'keras_models/' + dataset)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = dataset + '.h5'

    if dataset == "mnist":
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif dataset == "imagenet_inceptionv3":
        model = keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3), input_tensor=None, pooling=None)
        model_name_v1 = 'imagenet_inceptionV3_v1.hdf5'
        model_name_v2 = 'imagenet_inceptionV3_v2.hdf5'
        model_path = os.path.join(save_dir, model_name_v1)
        model.save(model_path)
        with h5py.File(model_path) as h5:
            config = json.loads(h5.attrs.get("model_config").decode('utf-8').replace('input_dtype', 'dtype'))
        with tf.Session('') as sess:
            model = model_from_config(config)
            model.load_weights(model_path)
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
            model_path = os.path.join(save_dir, model_name_v2)
            model.save(model_path)
            del model
        del sess
        print("InceptionV3 Model has been successfully downloaded and saved.")
        exit()
    elif dataset == "imagenet_xception":
        model = keras.applications.xception.Xception(include_top=True, weights='imagenet', input_shape=(299, 299, 3), input_tensor=None, pooling=None)
        model_name_v1 = 'imagenet_xception_v1.hdf5'
        model_name_v2 = 'imagenet_xception_v2.hdf5'
        model_path = os.path.join(save_dir, model_name_v1)
        model.save(model_path)
        with h5py.File(model_path) as h5:
            config = json.loads(h5.attrs.get("model_config").decode('utf-8').replace('input_dtype', 'dtype'))
        with tf.Session('') as sess:
            model = model_from_config(config)
            model.load_weights(model_path)
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])
            model_path = os.path.join(save_dir, model_name_v2)
            model.save(model_path)
            del model
        del sess
        print("Xception Model has been successfully downloaded and saved.")
        exit()
    else:
        print("Unknown dataset/Model!")
        exit()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    if dataset == "mnist":
        #LeNet Architecture
        model = keras.Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
        model.add(AveragePooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(AveragePooling2D())

        model.add(Flatten())
        model.add(Dense(units=120, activation='relu'))
        model.add(Dense(units=84, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.99, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, shuffle=True, batch_size=batch_size)

    # Save model and weights
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model on train set.
    scores = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss:', scores[0])
    print('Train accuracy:', scores[1])

    # Score trained model on test set.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
