#experiment with different setting and save in different epochs
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from extra_models.resnet import lr_schedule, resnet_v1, resnet_v2
import extra_models.densenet as densenet
import argparse
import h5py
import json
import os
import numpy as np
import tensorflow as tf
from keras.models import model_from_config


def lr_schedule_densenet(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


parser = argparse.ArgumentParser(description='Train a target model.')
parser.add_argument('-d', '--dataset', type=str, default='cifar_10', choices=['mnist', 'cifar_10', 'cifar_100', 'cifar_100_resnet', 'cifar_100_densenet', 'imagenet_inceptionv3', 'imagenet_xception'],  help='Indicate dataset and target model archtecture.')
parser.add_argument('-c', '--conv_blocks', type=int, default=0, help='The number of conv blocks for CIFAR10 and CIFAR100.')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate.')
parser.add_argument('-P', '--periodical_save', default=False, help='Save the model periodically during training.', action='store_true')
parser.add_argument('-p', '--save_period', type=int, default=10, help='Save the model each p epoch.')
parser.add_argument('-R', '--lr_reducer', default=False, help='Use learning reducer.', action='store_true')
parser.add_argument('-S', '--lr_scheduler', default=False, help='Use learning scheduler.', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset
    batch_size = args.batch_size
    use_lr_reducer = args.lr_reducer
    use_lr_scheduler = args.lr_scheduler
    periodic_save = args.periodical_save
    saving_period = args.save_period
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    save_dir_intermediate = os.path.join(save_dir, 'intermediate')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(save_dir_intermediate):
        os.makedirs(save_dir_intermediate)
    sub_model_name = dataset + '_weights_'
    model_name = sub_model_name + 'final.h5'

    if dataset == "mnist":
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif dataset == "cifar_10":
        num_classes = 10
        if args.conv_blocks <= 0:
            conv_blocks = 2
        elif args.conv_blocks < 4:
            conv_blocks = args.conv_blocks
        else:
            print("Error: The number of convolutional blocks shoud be 0, 1, 2, 3 or 4! (0 indicates default value for the model)")
            exit()
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "cifar_100":
        num_classes = 100
        if args.conv_blocks <= 0:
            conv_blocks = 3
        elif args.conv_blocks < 4:
            conv_blocks = args.conv_blocks
        else:
            print("Error: The number of convolutional blocks shoud be 0, 1, 2, 3 or 4! (0 indicates default value for the model)")
            exit()
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == "cifar_100":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == "cifar_100_resnet":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == "cifar_100_densenet":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    elif dataset == "cifar_100_inceptionv3":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
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
    elif dataset == "cifar_10" or dataset == "cifar_100":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        if conv_blocks >= 2:
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

        if conv_blocks >= 3:
            model.add(Conv2D(128, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

        if conv_blocks >= 4:
            model.add(Conv2D(256, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

        if dataset == "cifar_10":
            model.add(Flatten())
            model.add(Dense(128))
        elif dataset == "cifar_100":
            model.add(Flatten())
            model.add(Dense(1024))

        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    elif dataset == "cifar_100_resnet":
        # model = keras.applications.resnet.ResNet101(include_top=True, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=num_classes)
        #model = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=num_classes)
        model = resnet_v2(input_shape=x_train.shape[1:], depth=101, num_classes=num_classes)
    elif dataset == "cifar_100_densenet":
        #model = keras.applications.densenet.DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=num_classes)
        depth = 100
        nb_dense_block = 3
        growth_rate = 12
        nb_filter = 12
        bottleneck = False
        reduction = 0.0
        dropout_rate = 0.2  # 0.0 for data augmentation
        model = densenet.DenseNet((32, 32, 3), classes=num_classes, depth=70, nb_dense_block=nb_dense_block,
                              growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                              bottleneck=bottleneck, reduction=reduction, weights=None)

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.99, epsilon=1e-08)

    callbacks = []
    if periodic_save:
        filepath = save_dir_intermediate + "/" + sub_model_name + "_{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor=['val_accuracy'], verbose=1, save_best_only=False,
                                     mode='max', save_weights_only=False, period=saving_period)
        callbacks.append(checkpoint)

    if use_lr_reducer:
        lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks.append(lr_reducer)

    if use_lr_scheduler:
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks.append(lr_scheduler)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, shuffle=True, batch_size=batch_size,
                            callbacks=callbacks, use_multiprocessing=True)

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
