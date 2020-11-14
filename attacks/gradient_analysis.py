from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.datasets import cifar10, cifar100
import os
import numpy as np
from scipy.stats import norm, kurtosis, skew
from utils import average_over_gradient_metrics, wigthed_average, wigthed_average_over_gradient_metrics
from pandas.core.common import flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
reload_session_period = 40

show_correct_gradients = True
show_incorrect_gradients = True
save_gradients = True


def normalize_data(data, means, stds):
    data /= 255
    data[:, :, :, 0] -= means[0]
    data[:, :, :, 0] /= stds[0]
    data[:, :, :, 1] -= means[1]
    data[:, :, :, 1] /= stds[1]
    data[:, :, :, 2] -= means[2]
    data[:, :, :, 2] /= stds[2]
    return data


def reload_session(model_name):
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_name)
    return model


def gradient_based_attack_wrt_x(model, input, trg, num_classes):
    target = tf.keras.utils.to_categorical(trg, num_classes)
    loss = tf.keras.categorical_crossentropy(target, model.output)
    # loss = K.sum(K.square(model.output - target))
    # loss = tf.keras.backend.categorical_crossentropy(target, model.output)
    gradients = tf.keras.gradients(loss, model.input)[0]
    fn = tf.keras.function([model.input], [gradients])
    grads = fn([input])
    g = np.array((grads[0]))
    weight_grad = list(flatten(g))
    return return_norm_metrics(weight_grad)


def gradient_based_attack_wrt_w(model, inputs, trg, num_classes):
    output = tf.keras.utils.to_categorical(trg, num_classes)
    """ Gets gradient of model for given inputs and outputs for all weights"""
    # from: https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = tf.keras.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, output.reshape((1, num_classes)))
    weight_grad = f(x + y + sample_weight)
    weight_grad = list(flatten(weight_grad))
    # print(len(weight_grad))
    return return_norm_metrics(weight_grad)


def gradient_based_attack_wrt_x_batch(model, inputs, trg, num_classes):
    output = tf.keras.utils.to_categorical(trg, num_classes)
    output = output.reshape((1, num_classes))
    """ Gets gradient of model for given inputs and outputs for all weights"""
    # from: https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    grads = model.optimizer.get_gradients(model.total_loss, model.input)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = tf.keras.backend.function(symb_inputs, grads)
    norms = np.zeros((inputs.shape[0], 7))
    for i in range(inputs.shape[0]):
        x, y, sample_weight = model._standardize_user_data(inputs[i:i+1], output)
        x_grad = f(x + y + sample_weight)
        x_grad = list(flatten(x_grad))
        norms[i] = return_norm_metrics(x_grad)
        if i % reload_session_period == 0:
            print(i, norms[i])
    return norms


def gradient_based_attack_wrt_w_batch(model, inputs, trg, num_classes):
    output = tf.keras.utils.to_categorical(trg, num_classes)
    output = output.reshape((1, num_classes))
    """ Gets gradient of model for given inputs and outputs for all weights"""
    # from: https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = tf.keras.backend.function(symb_inputs, grads)
    norms = np.zeros((inputs.shape[0], 7))
    for i in range(inputs.shape[0]):
        x, y, sample_weight = model._standardize_user_data(inputs[i:i+1], output)
        weight_grad = f(x + y + sample_weight)
        weight_grad = list(flatten(weight_grad))
        norms[i] = return_norm_metrics(weight_grad)
        if i % reload_session_period == 0:
            print(i, norms[i])
    return norms


def return_norm_metrics(gradient):
    l1 = np.linalg.norm(gradient, ord=1)
    l2 = np.linalg.norm(gradient)
    Min = np.linalg.norm(gradient, ord=-np.inf)
    Max = np.linalg.norm(gradient, ord=np.inf)
    Mean = np.average(gradient)
    Skewness = skew(gradient)
    Kurtosis = kurtosis(gradient)
    return [l1, l2, Min, Max, Mean, Skewness, Kurtosis]


def gradient_norms(dataset, num_classes, num_targeted_classes, num_of_samples_per_class, model_name, gradient_save_dir):
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if dataset == "mnist":
        x_train /= 255
        x_test /= 255
    else:
        x_train = normalize_data(x_train, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        x_test = normalize_data(x_test, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        x_train = np.transpose(x_train, axes=(0, 3, 1, 2))
        x_test = np.transpose(x_test, axes=(0, 3, 1, 2))

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    print(model_name)
    model = tf.keras.models.load_model(model_name)

    confidence_train = model.predict(x_train)
    confidence_test = model.predict(x_test)
    labels_train_by_model = np.argmax(confidence_train, axis=1)
    labels_test_by_model = np.argmax(confidence_test, axis=1)
    labels_train = np.argmax(y_train, axis=1)
    labels_test = np.argmax(y_test, axis=1)

    correctly_classified_indexes_train = labels_train_by_model == labels_train
    incorrectly_classified_indexes_train = labels_train_by_model != labels_train

    correctly_classified_indexes_test = labels_test_by_model == labels_test
    incorrectly_classified_indexes_test = labels_test_by_model != labels_test

    # Contains 7 norms: L1/L2/Min/Max/Mean/Skewness/Kurtosis
    gradient_norm_wrt_x_correct_train = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_correct_train_std = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_correct_test = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_correct_test_std = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_incorrect_train = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_incorrect_train_std = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_incorrect_test = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_x_incorrect_test_std = np.zeros((num_targeted_classes, 7)) - 1

    # Contains 7 norms: L1/L2/Min/Max/Mean/Skewness/Kurtosis
    gradient_norm_wrt_w_correct_train = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_correct_train_std = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_correct_test = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_correct_test_std = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_incorrect_train = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_incorrect_train_std = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_incorrect_test = np.zeros((num_targeted_classes, 7)) - 1
    gradient_norm_wrt_w_incorrect_test_std = np.zeros((num_targeted_classes, 7)) - 1

    correct_train_samples = np.zeros(num_targeted_classes) - 1
    correct_test_samples = np.zeros(num_targeted_classes) - 1
    incorrect_train_samples = np.zeros(num_targeted_classes) - 1
    incorrect_test_samples = np.zeros(num_targeted_classes) - 1

    for j in range(num_targeted_classes):
        if show_correct_gradients:
            print('Computing gradient norms for class ', j)
            correctly_classified_indexes_train_of_this_class = np.logical_and(correctly_classified_indexes_train, labels_train == j)
            correctly_classified_indexes_test_of_this_class = np.logical_and(correctly_classified_indexes_test, labels_test == j)
            cor_class_yes_x = x_train[correctly_classified_indexes_train_of_this_class]
            cor_class_no_x = x_test[correctly_classified_indexes_test_of_this_class]

            cor_class_yes_indexes = np.nonzero(correctly_classified_indexes_train_of_this_class)[0]
            cor_class_no_indexes = np.nonzero(correctly_classified_indexes_test_of_this_class)[0]

            if num_of_samples_per_class > 0:
                if cor_class_yes_x.shape[0] > num_of_samples_per_class:
                    cor_class_yes_x = cor_class_yes_x[:num_of_samples_per_class]
                    cor_class_yes_indexes = cor_class_yes_indexes[:num_of_samples_per_class]
                if cor_class_no_x.shape[0] > num_of_samples_per_class:
                    cor_class_no_x = cor_class_no_x[:num_of_samples_per_class]
                    cor_class_no_indexes = cor_class_no_indexes[:num_of_samples_per_class]


            gradient_wrt_w_per_sample_train = gradient_based_attack_wrt_w_batch(model, cor_class_yes_x, j, num_classes)
            gradient_wrt_x_per_sample_train = gradient_based_attack_wrt_x_batch(model, cor_class_yes_x, j, num_classes)
            gradient_wrt_w_per_sample_test = gradient_based_attack_wrt_w_batch(model, cor_class_no_x, j, num_classes)
            gradient_wrt_x_per_sample_test = gradient_based_attack_wrt_x_batch(model, cor_class_no_x, j, num_classes)

            model = reload_session(model_name)

            if save_gradients:
                np.savez(gradient_save_dir + 'cor-w-train-' + str(j), gradient_wrt_w_per_sample_train, cor_class_yes_indexes)
                np.savez(gradient_save_dir + 'cor-w-test-' + str(j), gradient_wrt_w_per_sample_test, cor_class_no_indexes)
                np.savez(gradient_save_dir + 'cor-x-train-' + str(j), gradient_wrt_x_per_sample_train, cor_class_yes_indexes)
                np.savez(gradient_save_dir + 'cor-x-test-' + str(j), gradient_wrt_x_per_sample_test, cor_class_no_indexes)

            gradient_norm_wrt_w_correct_train[j], gradient_norm_wrt_w_correct_train_std[j] = average_over_gradient_metrics(gradient_wrt_w_per_sample_train)
            gradient_norm_wrt_w_correct_test[j], gradient_norm_wrt_w_correct_test_std[j] = average_over_gradient_metrics(gradient_wrt_w_per_sample_test)
            gradient_norm_wrt_x_correct_train[j], gradient_norm_wrt_x_correct_train_std[j] = average_over_gradient_metrics(gradient_wrt_x_per_sample_train)
            gradient_norm_wrt_x_correct_test[j], gradient_norm_wrt_x_correct_test_std[j] = average_over_gradient_metrics(gradient_wrt_x_per_sample_test)

            correct_train_samples[j] = gradient_wrt_w_per_sample_train.shape[1]
            correct_test_samples[j] = gradient_wrt_w_per_sample_test.shape[1]
            #print(correct_train_samples[j], correct_test_samples[j])
            #print(gradient_norm_wrt_w_correct_train[j], gradient_norm_wrt_w_correct_train_std[j])
            #print(gradient_norm_wrt_w_correct_test[j], gradient_norm_wrt_w_correct_test_std[j])
            #print(gradient_norm_wrt_x_correct_train[j], gradient_norm_wrt_x_correct_train_std[j])
            #print(gradient_norm_wrt_x_correct_test[j], gradient_norm_wrt_x_correct_test_std[j])

        if show_incorrect_gradients:
            print("incorrectly classified:")
            incorrectly_classified_indexes_train_of_this_class = np.logical_and(incorrectly_classified_indexes_train, labels_train == j)
            incorrectly_classified_indexes_test_of_this_class = np.logical_and(incorrectly_classified_indexes_test, labels_test == j)
            incor_class_yes_x = x_train[incorrectly_classified_indexes_train_of_this_class]
            incor_class_no_x = x_test[incorrectly_classified_indexes_test_of_this_class]
            incor_class_yes_indexes = np.nonzero(incorrectly_classified_indexes_train_of_this_class)[0]
            incor_class_no_indexes = np.nonzero(incorrectly_classified_indexes_test_of_this_class)[0]

            if incor_class_yes_x.shape[0] < 5 or incor_class_no_x.shape[0] < 5:
                print("skip distance computation for inccorectly labeled samples due to lack os misclassified samples!")
            else:
                if num_of_samples_per_class > 0:
                    if incor_class_yes_x.shape[0] > num_of_samples_per_class:
                        incor_class_yes_x = incor_class_yes_x[:num_of_samples_per_class]
                        incor_class_yes_indexes = incor_class_yes_indexes[:num_of_samples_per_class]
                    if incor_class_no_x.shape[0] > num_of_samples_per_class:
                        incor_class_no_x = incor_class_no_x[:num_of_samples_per_class]
                        incor_class_no_indexes = incor_class_no_indexes[:num_of_samples_per_class]


                gradient_wrt_w_per_sample_train = gradient_based_attack_wrt_w_batch(model, incor_class_yes_x, j, num_classes)
                gradient_wrt_x_per_sample_train = gradient_based_attack_wrt_x_batch(model, incor_class_yes_x, j, num_classes)
                gradient_wrt_w_per_sample_test = gradient_based_attack_wrt_w_batch(model, incor_class_no_x, j, num_classes)
                gradient_wrt_x_per_sample_test = gradient_based_attack_wrt_x_batch(model, incor_class_no_x, j, num_classes)


                model = reload_session(model_name)

                if save_gradients:
                    np.savez(gradient_save_dir + 'incor-w-train-' + str(j), gradient_wrt_w_per_sample_train, incor_class_yes_indexes)
                    np.savez(gradient_save_dir + 'incor-w-test-' + str(j), gradient_wrt_w_per_sample_test, incor_class_no_indexes)
                    np.savez(gradient_save_dir + 'incor-x-train-' + str(j), gradient_wrt_x_per_sample_train, incor_class_yes_indexes)
                    np.savez(gradient_save_dir + 'incor-x-test-' + str(j), gradient_wrt_x_per_sample_test, incor_class_no_indexes)


                gradient_norm_wrt_w_incorrect_train[j], gradient_norm_wrt_w_incorrect_train_std[j] = average_over_gradient_metrics(gradient_wrt_w_per_sample_train)
                gradient_norm_wrt_w_incorrect_test[j], gradient_norm_wrt_w_incorrect_test_std[j] = average_over_gradient_metrics(gradient_wrt_w_per_sample_test)
                gradient_norm_wrt_x_incorrect_train[j], gradient_norm_wrt_x_incorrect_train_std[j] = average_over_gradient_metrics(gradient_wrt_x_per_sample_train)
                gradient_norm_wrt_x_incorrect_test[j], gradient_norm_wrt_x_incorrect_test_std[j] = average_over_gradient_metrics(gradient_wrt_x_per_sample_test)

                incorrect_train_samples[j] = gradient_wrt_w_per_sample_train.shape[1]
                incorrect_test_samples[j] = gradient_wrt_w_per_sample_test.shape[1]
                #print(incorrect_train_samples[j], incorrect_test_samples[j])
                #print(gradient_norm_wrt_w_incorrect_train[j], gradient_norm_wrt_w_incorrect_train_std[j])
                #print(gradient_norm_wrt_w_incorrect_test[j], gradient_norm_wrt_w_incorrect_test_std[j])
                #print(gradient_norm_wrt_x_incorrect_train[j], gradient_norm_wrt_x_incorrect_train_std[j])
                #print(gradient_norm_wrt_x_incorrect_test[j], gradient_norm_wrt_x_incorrect_test_std[j])

        print("class ", str(j), " is finished.")

    avg_w_correct_train = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_correct_train, correct_train_samples)
    avg_w_correct_train_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_correct_train_std, correct_train_samples)
    avg_w_correct_test = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_correct_test, correct_test_samples)
    avg_w_correct_test_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_correct_test_std, correct_test_samples)
    avg_x_correct_train = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_correct_train, correct_train_samples)
    avg_x_correct_train_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_correct_train_std, correct_train_samples)
    avg_x_correct_test = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_correct_test, correct_test_samples)
    avg_x_correct_test_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_correct_test_std, correct_test_samples)

    avg_w_incorrect_train = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_incorrect_train, incorrect_train_samples)
    avg_w_incorrect_train_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_incorrect_train_std, incorrect_train_samples)
    avg_w_incorrect_test = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_incorrect_test, incorrect_test_samples)
    avg_w_incorrect_test_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_w_incorrect_test_std, incorrect_test_samples)
    avg_x_incorrect_train = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_incorrect_train, incorrect_train_samples)
    avg_x_incorrect_train_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_incorrect_train_std, incorrect_train_samples)
    avg_x_incorrect_test = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_incorrect_test, incorrect_test_samples)
    avg_x_incorrect_test_std = wigthed_average_over_gradient_metrics(gradient_norm_wrt_x_incorrect_test_std, incorrect_test_samples)

    print("\n\nFinal Results:")
    if show_correct_gradients:
        print("Correctly labeled (wrt w):")
        print("Train set: ", avg_w_correct_train, avg_w_correct_train_std)
        print("Test set: ", avg_w_correct_test, avg_w_correct_test_std)

        print("\nCorrectly labeled (wrt x):")
        print("Train set: ", avg_x_correct_train, avg_x_correct_train_std)
        print("Test set: ", avg_x_correct_test, avg_x_correct_test_std)


    if show_incorrect_gradients:
        print("\nIncorrectly labeled (wrt w):")
        print("Train set: ", avg_w_incorrect_train, avg_w_incorrect_train_std)
        print("Test set: ", avg_w_incorrect_test, avg_w_incorrect_test_std)

        print("\nIncorrectly labeled (wrt x):")
        print("Train set: ", avg_x_incorrect_train, avg_x_incorrect_train_std)
        print("Test set: ", avg_x_incorrect_test, avg_x_incorrect_test_std)
