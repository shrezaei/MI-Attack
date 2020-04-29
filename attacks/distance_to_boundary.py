from __future__ import print_function
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.datasets import cifar10, cifar100
import numpy as np
from utils import average_over_positive_values, wigthed_average
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
reload_session_period = 20
show_correct_distance = True
show_incorrect_distance = True
save_distances = True


def reload_session(model_name):
    keras.backend.clear_session()
    model = keras.models.load_model(model_name)
    return model


#Return the confidence difference between closet maximum and the current x
#It is untargeted FGSM, so trg refer to the class of the original input
def dist_to_boundary(input, model, trg, eps_step=0.001, eps=0.1, fgsm_max_steps=8000, norm=np.inf, boundary_steps=500, num_classes=100):
    target = keras.utils.to_categorical(trg, num_classes)
    loss = K.sum(K.square(model.output - target))
    gradients = K.gradients(loss, model.input)[0]
    fn = K.function([model.input], [gradients])
    adv_x = input
    l2_distance = -1
    for i in range(fgsm_max_steps):
        if i % 10 == 0:
            grads = fn([adv_x])
            g = np.array((grads[0]))
            if norm == np.inf:
                grad = np.sign(g)
            elif norm == 2:
                l2_grad = (np.sqrt(np.sum(np.square(g), axis=0, keepdims=True)))
                # To avoid division by zero
                l2_grad += 10e-12
                grad = g / l2_grad
        adv_x += eps_step * grad
        adv_x = np.clip(adv_x, 0, 1)
        adv_conf = model.predict(adv_x)
        adv_label = np.argmax(adv_conf[0])
        if adv_label != trg:
            adv_conf = model.predict(adv_x)
            adv_label = np.argmax(adv_conf[0])
            # adv_x and temp_x are the samples that lay on difference sides of a boundary.
            temp_x = adv_x - eps_step * grad
            temp_x = np.clip(temp_x, 0, 1)
            temp_conf = model.predict(temp_x)
            label3 = np.argmax(temp_conf[0])
            conf_difference = np.abs(adv_conf[0, adv_label] - temp_conf[0, label3])
            # This while tries to find the a closer point to the boundary as much as possible
            conf_difference_threshold = 0.0001
            j = 0
            intermediate_x = adv_x
            while conf_difference > conf_difference_threshold and j < boundary_steps:
                j += 1
                intermediate_x = (adv_x + temp_x) / 2
                intermediate_conf = model.predict(intermediate_x)
                intermediate_label = np.argmax(intermediate_conf[0])
                if intermediate_label == label3:
                    temp_x = intermediate_x
                    temp_conf = intermediate_conf
                    label3 = intermediate_label
                elif intermediate_label == adv_label:
                    adv_x = intermediate_x
                    adv_conf = intermediate_conf
                    adv_label = intermediate_label
                else:
                    # If this happens, it probably means that boundary, in small region around closet
                    # boundary of the input sample is not a simple linear function between two classes
                    # print("Warning: Unexpected boundary region.", adv_label, label3, intermediate_label)
                    temp_x = intermediate_x
                    temp_conf = intermediate_conf
                    label3 = intermediate_label
                conf_difference = np.abs(adv_conf[0, adv_label] - temp_conf[0, label3])
            l2_distance = np.linalg.norm(intermediate_x - input)
            break
    return l2_distance


def distance_to_boundary(dataset, num_classes, num_targeted_classes, num_of_samples_per_class, model_name, distance_save_dir):
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    elif dataset == "cifar_10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    print(model_name)
    model = keras.models.load_model(model_name)

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

    distance_correct_train = np.zeros(num_targeted_classes) - 1
    distance_correct_train_std = np.zeros(num_targeted_classes) - 1
    distance_correct_test = np.zeros(num_targeted_classes) - 1
    distance_correct_test_std = np.zeros(num_targeted_classes) - 1
    distance_incorrect_train = np.zeros(num_targeted_classes) - 1
    distance_incorrect_train_std = np.zeros(num_targeted_classes) - 1
    distance_incorrect_test = np.zeros(num_targeted_classes) - 1
    distance_incorrect_test_std = np.zeros(num_targeted_classes) - 1

    correct_train_samples = np.zeros(num_targeted_classes) - 1
    correct_test_samples = np.zeros(num_targeted_classes) - 1
    incorrect_train_samples = np.zeros(num_targeted_classes) - 1
    incorrect_test_samples = np.zeros(num_targeted_classes) - 1

    for j in range(1, num_targeted_classes):

        if show_correct_distance:
            print('Computing distance for class ', j)
            correctly_classified_indexes_train_of_this_class = np.logical_and(correctly_classified_indexes_train, labels_train == j)
            correctly_classified_indexes_test_of_this_class = np.logical_and(correctly_classified_indexes_test, labels_test == j)
            cor_class_yes_x = x_train[correctly_classified_indexes_train_of_this_class]
            cor_class_no_x = x_test[correctly_classified_indexes_test_of_this_class]

            if num_of_samples_per_class > 0:
                if cor_class_yes_x.shape[0] > num_of_samples_per_class:
                    cor_class_yes_x = cor_class_yes_x[:num_of_samples_per_class]
                if cor_class_no_x.shape[0] > num_of_samples_per_class:
                    cor_class_no_x = cor_class_no_x[:num_of_samples_per_class]

            distance_per_sample_train = np.zeros(cor_class_yes_x.shape[0]) - 1
            distance_per_sample_test = np.zeros(cor_class_no_x.shape[0]) - 1
            for i in range(cor_class_yes_x.shape[0]):
                distance_per_sample_train[i] = dist_to_boundary(cor_class_yes_x[i:i+1], model, j, norm=2, num_classes=num_classes)
                if i % reload_session_period == 0:
                    model = reload_session(model_name)
                    print('Train samples progress: ', i, '/', cor_class_yes_x.shape[0])
            for i in range(cor_class_no_x.shape[0]):
                distance_per_sample_test[i] = dist_to_boundary(cor_class_no_x[i:i+1], model, j, norm=2, num_classes=num_classes)
                if i % reload_session_period == 0:
                    model = reload_session(model_name)
                    print('Test samples progress: ', i, '/', cor_class_no_x.shape[0])

            if save_distances:
                np.save(distance_save_dir + '/' + model_name.split('/')[-1] + '-cor-train-' + str(j), distance_per_sample_train)
                np.save(distance_save_dir + '/' + model_name.split('/')[-1] + '-cor-test-' + str(j), distance_per_sample_test)

            distance_per_sample_train = distance_per_sample_train[distance_per_sample_train != -1]
            distance_per_sample_test = distance_per_sample_test[distance_per_sample_test != -1]

            distance_correct_train[j], distance_correct_train_std[j] = average_over_positive_values(distance_per_sample_train)
            distance_correct_test[j], distance_correct_test_std[j] = average_over_positive_values(distance_per_sample_test)

            correct_train_samples[j] = distance_per_sample_train.shape[0]
            correct_test_samples[j] = distance_per_sample_test.shape[0]


        if show_incorrect_distance:
            print("incorrectly classified...")
            incorrectly_classified_indexes_train_of_this_class = np.logical_and(incorrectly_classified_indexes_train, labels_train == j)
            incorrectly_classified_indexes_test_of_this_class = np.logical_and(incorrectly_classified_indexes_test, labels_test == j)
            incor_class_yes_x = x_train[incorrectly_classified_indexes_train_of_this_class]
            incor_class_no_x = x_test[incorrectly_classified_indexes_test_of_this_class]

            if incor_class_yes_x.shape[0] < 10 or incor_class_no_x.shape[0] < 10:
                print("skip distance computation for inccorectly labeled samples due to lack os misclassified samples!")
            else:
                if num_of_samples_per_class > 0:
                    if incor_class_yes_x.shape[0] > num_of_samples_per_class:
                        incor_class_yes_x = incor_class_yes_x[:num_of_samples_per_class]
                    if incor_class_no_x.shape[0] > num_of_samples_per_class:
                        incor_class_no_x = incor_class_no_x[:num_of_samples_per_class]

                distance_per_sample_train = np.zeros(incor_class_yes_x.shape[0]) - 1
                distance_per_sample_test = np.zeros(incor_class_no_x.shape[0]) - 1
                print(distance_per_sample_train.shape, distance_per_sample_test.shape)
                for i in range(incor_class_yes_x.shape[0]):
                    distance_per_sample_train[i] = dist_to_boundary(incor_class_yes_x[i:i+1], model, j, norm=2, num_classes=num_classes)
                    if i % reload_session_period == 0:
                        model = reload_session(model_name)
                        print('Train samples progress: ', i, '/', incor_class_yes_x.shape[0])
                for i in range(incor_class_no_x.shape[0]):
                    distance_per_sample_test[i] = dist_to_boundary(incor_class_no_x[i:i+1], model, j, norm=2, num_classes=num_classes)
                    if i % reload_session_period == 0:
                        model = reload_session(model_name)
                        print('Train samples progress: ', i, '/', incor_class_no_x.shape[0])

                if save_distances:
                    np.save(distance_save_dir + '/' + model_name.split('/')[-1] + '-incor-train-' + str(j), distance_per_sample_train)
                    np.save(distance_save_dir + '/' + model_name.split('/')[-1] + '-incor-test-' + str(j), distance_per_sample_test)

                distance_per_sample_train = distance_per_sample_train[distance_per_sample_train != -1]
                distance_per_sample_test = distance_per_sample_test[distance_per_sample_test != -1]

                distance_incorrect_train[j], distance_incorrect_train_std[j] = average_over_positive_values(distance_per_sample_train)
                distance_incorrect_test[j], distance_incorrect_test_std[j] = average_over_positive_values(distance_per_sample_test)

                incorrect_train_samples[j] = distance_per_sample_train.shape[0]
                incorrect_test_samples[j] = distance_per_sample_test.shape[0]

        avg_correct_train = wigthed_average(distance_correct_train, correct_train_samples)
        avg_correct_train_std = wigthed_average(distance_correct_train_std, correct_train_samples)
        avg_correct_test = wigthed_average(distance_correct_test, correct_test_samples)
        avg_correct_test_std = wigthed_average(distance_correct_test_std, correct_test_samples)

        avg_incorrect_train = wigthed_average(distance_incorrect_train, incorrect_train_samples)
        avg_incorrect_train_std = wigthed_average(distance_incorrect_train_std, incorrect_train_samples)
        avg_incorrect_test = wigthed_average(distance_incorrect_test, incorrect_test_samples)
        avg_incorrect_test_std = wigthed_average(distance_incorrect_test_std, incorrect_test_samples)
        print("\nResults up to class ", str(j), ":")
        if show_correct_distance:
            print("Correctly labeled:")
            print(avg_correct_train, avg_correct_train_std, avg_correct_test, avg_correct_test_std)

        if show_incorrect_distance:
            print("Incorrectly labeled:")
            print(avg_incorrect_train, avg_incorrect_train_std, avg_incorrect_test, avg_incorrect_test_std)


    avg_correct_train = wigthed_average(distance_correct_train, correct_train_samples)
    avg_correct_train_std = wigthed_average(distance_correct_train_std, correct_train_samples)
    avg_correct_test = wigthed_average(distance_correct_test, correct_test_samples)
    avg_correct_test_std = wigthed_average(distance_correct_test_std, correct_test_samples)

    avg_incorrect_train = wigthed_average(distance_incorrect_train, incorrect_train_samples)
    avg_incorrect_train_std = wigthed_average(distance_incorrect_train_std, incorrect_train_samples)
    avg_incorrect_test = wigthed_average(distance_incorrect_test, incorrect_test_samples)
    avg_incorrect_test_std = wigthed_average(distance_incorrect_test_std, incorrect_test_samples)

    print("\n\nFinal Results:")
    if show_correct_distance:
        print("Correctly labeled: [train_average train_standard_deviation test_average test_standard_deviation]")
        print(avg_correct_train, avg_correct_train_std, avg_correct_test, avg_correct_test_std)

    if show_incorrect_distance:
        print("Incorrectly labeled: [train_average train_standard_deviation test_average test_standard_deviation]")
        print(avg_incorrect_train, avg_incorrect_train_std, avg_incorrect_test, avg_incorrect_test_std)

