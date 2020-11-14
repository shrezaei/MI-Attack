from __future__ import print_function
import keras
from keras import backend as K
import os
import numpy as np
from scipy.stats import norm, kurtosis, skew
import pdb
from utils import average_over_positive_values, average_over_gradient_metrics, wigthed_average, load_Data_with_imagenet_id, wigthed_average_over_gradient_metrics
from pandas.core.common import flatten
from keras.models import model_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
progress_print_period = 20

show_correct_gradients = True
show_incorrect_gradients = True
save_gradients = True


def reload_session(model_name):
    keras.backend.clear_session()
    model = keras.models.load_model(model_name)
    return model


def gradient_based_attack_wrt_x(model, input, trg, num_classes):
    target = keras.utils.to_categorical(trg, num_classes)
    loss = K.categorical_crossentropy(target, model.output)
    # loss = K.sum(K.square(model.output - target))
    gradients = K.gradients(loss, model.input)[0]
    fn = K.function([model.input], [gradients])
    grads = fn([input])
    g = np.array((grads[0]))
    weight_grad = list(flatten(g))
    return return_norm_metrics(weight_grad)

def gradient_based_attack_wrt_x_batch(model, inputs, trg, num_classes):
    target = keras.utils.to_categorical(trg, num_classes)
    loss = K.categorical_crossentropy(target, model.output)
    # loss = K.sum(K.square(model.output - target))
    gradients = K.gradients(loss, model.input)[0]
    fn = K.function([model.input], [gradients])
    grads = fn([inputs])
    g = np.array((grads[0]))
    norms = np.zeros((inputs.shape[0], 7))
    for i in range(inputs.shape[0]):
        x_grad = list(flatten(g[i]))
        norms[i] = return_norm_metrics(x_grad)
        if i % progress_print_period == 0:
            print(i, norms[i])
    return norms


def gradient_based_attack_wrt_w(model, input, trg, num_classes):
    target = keras.utils.to_categorical(trg, num_classes)
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.trainable_weights)
    fn = K.function([model.input], grads)
    g = fn([input])
    weight_grad = list(flatten(g))
    return return_norm_metrics(weight_grad)


def gradient_based_attack_wrt_w_batch(model, inputs, trg, num_classes):
    target = keras.utils.to_categorical(trg, num_classes)
    output = target.reshape((1, num_classes))
    """ Gets gradient of model for given inputs and outputs for all weights"""
    # from: https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    # grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    norms = np.zeros((inputs.shape[0], 7))
    for i in range(inputs.shape[0]):
        x, y, sample_weight = model._standardize_user_data(inputs[i:i+1], output)
        weight_grad = f(x + sample_weight)
        weight_grad = list(flatten(weight_grad))
        norms[i] = return_norm_metrics(weight_grad)
        if i % progress_print_period == 0:
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


def gradient_norms_imagenet(dataset, num_classes, num_targeted_classes, num_of_samples_per_class, model_name, gradient_save_dir, imagenet_path):
    print(model_name)
    model = keras.models.load_model(model_name)

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

        (x_train, y_train), (x_test, y_test), keras_class_id = load_Data_with_imagenet_id(j+1, imagenet_path=imagenet_path)
        print(x_train.shape, y_train.shape)

        x_train = keras.applications.inception_v3.preprocess_input(x_train)

        temp = gradient_based_attack_wrt_w_batch(model, x_train[0:1], j, num_classes)

        x_test = keras.applications.inception_v3.preprocess_input(x_test)

        labels_train_by_model = model.predict(x_train)
        labels_test_by_model = model.predict(x_test)
        labels_train_by_model = np.argmax(labels_train_by_model, axis=1)
        labels_test_by_model = np.argmax(labels_test_by_model, axis=1)

        labels_train = y_train
        labels_test = y_test

        correctly_classified_indexes_train = labels_train_by_model == labels_train
        incorrectly_classified_indexes_train = labels_train_by_model != labels_train

        correctly_classified_indexes_test = labels_test_by_model == labels_test
        incorrectly_classified_indexes_test = labels_test_by_model != labels_test


        if show_correct_gradients:
            print('Computing gradient norms for class ', j)
            cor_class_yes_x = x_train[correctly_classified_indexes_train]
            cor_class_no_x = x_test[correctly_classified_indexes_test]

            cor_class_yes_indexes = np.nonzero(correctly_classified_indexes_train)[0]
            cor_class_no_indexes = np.nonzero(correctly_classified_indexes_test)[0]

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

            print(gradient_wrt_x_per_sample_train.shape, gradient_wrt_x_per_sample_test.shape)

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
            incor_class_yes_x = x_train[incorrectly_classified_indexes_train]
            incor_class_no_x = x_test[incorrectly_classified_indexes_test]
            incor_class_yes_indexes = np.nonzero(incorrectly_classified_indexes_train)[0]
            incor_class_no_indexes = np.nonzero(incorrectly_classified_indexes_test)[0]

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

        print("Class ", str(j), " is finished.")

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
