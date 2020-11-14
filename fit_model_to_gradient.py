from __future__ import print_function
import os
import os.path
import numpy as np
import pdb
import argparse
from utils import average_of_gradient_metrics_of_2d_array, average_of_gradient_metrics, wigthed_average_for_gradient_metrics, false_alarm_rate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score

show_correct_distance = True
show_incorrect_distance = True
index_available_in_the_saved_files = True

parser = argparse.ArgumentParser(description='MI attack besed on distance to the boundary.')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-p', '--save_path', type=str, default='saved_gradients/cifar10/alexnet', help='Indicate the directory that the computed distances are saved into.')
args = parser.parse_args()


def fit_logistic_regression_model(a, b, single_dimension=True):
    model = LogisticRegression(class_weight='balanced')

    n1 = a.shape[0]
    n2 = b.shape[0]
    train_size_a = int(a.shape[0] * 0.8)
    train_size_b = int(b.shape[0] * 0.8)
    train_x_a = a[:train_size_a]
    train_y_a = np.zeros(train_size_a)
    train_x_b = b[:train_size_b]
    train_y_b = np.ones(train_size_b)

    test_x_a = a[train_size_a:]
    test_y_a = np.zeros(n1 - train_size_a)
    test_x_b = b[train_size_b:]
    test_y_b = np.ones(n2 - train_size_b)

    x_train = np.concatenate((train_x_a, train_x_b))
    y_train = np.concatenate((train_y_a, train_y_b))
    x_test = np.concatenate((test_x_a, test_x_b))
    y_test = np.concatenate((test_y_a, test_y_b))

    if single_dimension:
        x_train = x_train.reshape((-1, 1))
        x_test = x_test.reshape((-1, 1))

    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)
    results = balanced_accuracy_score(y_train, y_pred)
    print("train accu: ", results)

    y_pred = model.predict(x_test)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    far = false_alarm_rate(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    return bal_accuracy, accuracy, far, precision, recall, f1

if __name__ == '__main__':
    dataset = args.dataset
    gradient_saved_directory = args.save_path + '/'

    if dataset == "mnist" or dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000
    else:
        print("Unknown dataset!")
        exit()


    gradient_wrt_w_correct_train = np.zeros((num_classes, 7)) - 1
    gradient_wrt_w_correct_train_std = np.zeros((num_classes, 7)) - 1
    gradient_wrt_w_correct_test = np.zeros((num_classes, 7)) - 1
    gradient_wrt_w_correct_test_std = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_correct_train = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_correct_train_std = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_correct_test = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_correct_test_std = np.zeros((num_classes, 7)) - 1

    gradient_wrt_w_incorrect_train = np.zeros((num_classes, 7)) - 1
    gradient_wrt_w_incorrect_train_std = np.zeros((num_classes, 7)) - 1
    gradient_wrt_w_incorrect_test = np.zeros((num_classes, 7)) - 1
    gradient_wrt_w_incorrect_test_std = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_incorrect_train = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_incorrect_train_std = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_incorrect_test = np.zeros((num_classes, 7)) - 1
    gradient_wrt_x_incorrect_test_std = np.zeros((num_classes, 7)) - 1

    correct_train_samples = np.zeros(num_classes) - 1
    correct_test_samples = np.zeros(num_classes) - 1
    incorrect_train_samples = np.zeros(num_classes) - 1
    incorrect_test_samples = np.zeros(num_classes) - 1


    # To store per-class MI attack accuracy
    # 8 elemnts of the arrays are for the model trained on all 7 metrics simeltaneously
    bal_acc_per_class_w_correctly_labeled = np.zeros((num_classes, 8)) - 1
    bal_acc_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8)) - 1
    bal_acc_per_class_x_correctly_labeled = np.zeros((num_classes, 8)) - 1
    bal_acc_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8)) - 1

    acc_per_class_w_correctly_labeled = np.zeros((num_classes, 8)) - 1
    acc_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8)) - 1
    acc_per_class_x_correctly_labeled = np.zeros((num_classes, 8)) - 1
    acc_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8)) - 1

    far_per_class_w_correctly_labeled = np.zeros((num_classes, 8)) - 1
    far_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8)) - 1
    far_per_class_x_correctly_labeled = np.zeros((num_classes, 8)) - 1
    far_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8)) - 1

    prec_per_class_w_correctly_labeled = np.zeros((num_classes, 8, 2)) - 1
    prec_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8, 2)) - 1
    prec_per_class_x_correctly_labeled = np.zeros((num_classes, 8, 2)) - 1
    prec_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8, 2)) - 1

    rcal_per_class_w_correctly_labeled = np.zeros((num_classes, 8, 2)) - 1
    rcal_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8, 2)) - 1
    rcal_per_class_x_correctly_labeled = np.zeros((num_classes, 8, 2)) - 1
    rcal_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8, 2)) - 1

    f1_per_class_w_correctly_labeled = np.zeros((num_classes, 8, 2)) - 1
    f1_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8, 2)) - 1
    f1_per_class_x_correctly_labeled = np.zeros((num_classes, 8, 2)) - 1
    f1_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8, 2)) - 1

    for j in range(num_classes):

        if show_correct_distance:
            if index_available_in_the_saved_files:
                train_data_file_w = gradient_saved_directory + 'cor-w-train-' + str(j) + '.npz'
                test_data_file_w = gradient_saved_directory + 'cor-w-test-' + str(j) + '.npz'
                train_data_file_x = gradient_saved_directory + 'cor-x-train-' + str(j) + '.npz'
                test_data_file_x = gradient_saved_directory + 'cor-x-test-' + str(j) + '.npz'
                if os.path.isfile(train_data_file_w) and os.path.isfile(test_data_file_w) and os.path.isfile(
                        train_data_file_x) and os.path.isfile(test_data_file_x):
                    gradient_wrt_w_per_sample_train = np.nan_to_num(np.load(train_data_file_w)['arr_0'], posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_w_per_sample_test = np.nan_to_num(np.load(test_data_file_w)['arr_0'], posinf=100000,
                                                                   neginf=-100000)
                    gradient_wrt_x_per_sample_train = np.nan_to_num(np.load(train_data_file_x)['arr_0'], posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_x_per_sample_test = np.nan_to_num(np.load(test_data_file_x)['arr_0'], posinf=100000,
                                                                   neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for correctly labeled samples)!")
                    continue
            else:
                train_data_file_w = gradient_saved_directory + 'cor-w-train-' + str(j) + '.npy'
                test_data_file_w = gradient_saved_directory + 'cor-w-test-' + str(j) + '.npy'
                train_data_file_x = gradient_saved_directory + 'cor-x-train-' + str(j) + '.npy'
                test_data_file_x = gradient_saved_directory + 'cor-x-test-' + str(j) + '.npy'
                if os.path.isfile(train_data_file_w) and os.path.isfile(test_data_file_w) and os.path.isfile(
                        train_data_file_x) and os.path.isfile(test_data_file_x):
                    gradient_wrt_w_per_sample_train = np.nan_to_num(np.load(train_data_file_w), posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_w_per_sample_test = np.nan_to_num(np.load(test_data_file_w), posinf=100000,
                                                                   neginf=-100000)
                    gradient_wrt_x_per_sample_train = np.nan_to_num(np.load(train_data_file_x), posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_x_per_sample_test = np.nan_to_num(np.load(test_data_file_x), posinf=100000,
                                                                   neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for correctly labeled samples)!")
                    continue

            gradient_wrt_w_correct_train[j], gradient_wrt_w_correct_train_std[j] = average_of_gradient_metrics(
                gradient_wrt_w_per_sample_train)
            gradient_wrt_w_correct_test[j], gradient_wrt_w_correct_test_std[j] = average_of_gradient_metrics(
                gradient_wrt_w_per_sample_test)
            gradient_wrt_x_correct_train[j], gradient_wrt_x_correct_train_std[j] = average_of_gradient_metrics(
                gradient_wrt_x_per_sample_train)
            gradient_wrt_x_correct_test[j], gradient_wrt_x_correct_test_std[j] = average_of_gradient_metrics(
                gradient_wrt_x_per_sample_test)

            correct_train_samples[j] = gradient_wrt_w_per_sample_train.shape[0]
            correct_test_samples[j] = gradient_wrt_w_per_sample_test.shape[0]

            for i in range(7):
                print(i)
                bal_acc_per_class_w_correctly_labeled[j, i], acc_per_class_w_correctly_labeled[j, i], \
                far_per_class_w_correctly_labeled[j, i], prec_per_class_w_correctly_labeled[j, i], \
                rcal_per_class_w_correctly_labeled[j, i], \
                f1_per_class_w_correctly_labeled[j, i] = fit_logistic_regression_model(
                    gradient_wrt_w_per_sample_train[:, i], gradient_wrt_w_per_sample_test[:, i])

                bal_acc_per_class_x_correctly_labeled[j, i], acc_per_class_x_correctly_labeled[j, i], \
                far_per_class_x_correctly_labeled[j, i], prec_per_class_x_correctly_labeled[j, i], \
                rcal_per_class_x_correctly_labeled[j, i], \
                f1_per_class_x_correctly_labeled[j, i] = fit_logistic_regression_model(
                    gradient_wrt_x_per_sample_train[:, i], gradient_wrt_x_per_sample_test[:, i])

            bal_acc_per_class_w_correctly_labeled[j, 7], acc_per_class_w_correctly_labeled[j, 7], \
            far_per_class_w_correctly_labeled[j, 7], prec_per_class_w_correctly_labeled[j, 7], \
            rcal_per_class_w_correctly_labeled[j, 7], \
            f1_per_class_w_correctly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_w_per_sample_train,
                                                                                   gradient_wrt_w_per_sample_test,
                                                                                   single_dimension=False)

            bal_acc_per_class_x_correctly_labeled[j, 7], acc_per_class_x_correctly_labeled[j, 7], \
            far_per_class_x_correctly_labeled[j, 7], prec_per_class_x_correctly_labeled[j, 7], \
            rcal_per_class_x_correctly_labeled[j, 7], \
            f1_per_class_x_correctly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_x_per_sample_train,
                                                                                   gradient_wrt_x_per_sample_test,
                                                                                   single_dimension=False)

        if show_incorrect_distance:
            if index_available_in_the_saved_files:
                train_data_file_w = gradient_saved_directory + 'incor-w-train-' + str(j) + '.npz'
                test_data_file_w = gradient_saved_directory + 'incor-w-test-' + str(j) + '.npz'
                train_data_file_x = gradient_saved_directory + 'incor-x-train-' + str(j) + '.npz'
                test_data_file_x = gradient_saved_directory + 'incor-x-test-' + str(j) + '.npz'
                if os.path.isfile(train_data_file_w) and os.path.isfile(test_data_file_w) and os.path.isfile(
                        train_data_file_x) and os.path.isfile(test_data_file_x):
                    gradient_wrt_w_per_sample_train = np.nan_to_num(np.load(train_data_file_w)['arr_0'], posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_w_per_sample_test = np.nan_to_num(np.load(test_data_file_w)['arr_0'], posinf=100000,
                                                                   neginf=-100000)
                    gradient_wrt_x_per_sample_train = np.nan_to_num(np.load(train_data_file_x)['arr_0'], posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_x_per_sample_test = np.nan_to_num(np.load(test_data_file_x)['arr_0'], posinf=100000,
                                                                   neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for incorrectly labeled samples)!")
                    continue
            else:
                train_data_file_w = gradient_saved_directory + 'incor-w-train-' + str(j) + '.npy'
                test_data_file_w = gradient_saved_directory + 'incor-w-test-' + str(j) + '.npy'
                train_data_file_x = gradient_saved_directory + 'incor-x-train-' + str(j) + '.npy'
                test_data_file_x = gradient_saved_directory + 'incor-x-test-' + str(j) + '.npy'
                if os.path.isfile(train_data_file_w) and os.path.isfile(test_data_file_w) and os.path.isfile(
                        train_data_file_x) and os.path.isfile(test_data_file_x):
                    gradient_wrt_w_per_sample_train = np.nan_to_num(np.load(train_data_file_w), posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_w_per_sample_test = np.nan_to_num(np.load(test_data_file_w), posinf=100000,
                                                                   neginf=-100000)
                    gradient_wrt_x_per_sample_train = np.nan_to_num(np.load(train_data_file_x), posinf=100000,
                                                                    neginf=-100000)
                    gradient_wrt_x_per_sample_test = np.nan_to_num(np.load(test_data_file_x), posinf=100000,
                                                                   neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for incorrectly labeled samples)!")
                    continue

            gradient_wrt_w_incorrect_train[j], gradient_wrt_w_incorrect_train_std[j] = average_of_gradient_metrics(
                gradient_wrt_w_per_sample_train)
            gradient_wrt_w_incorrect_test[j], gradient_wrt_w_incorrect_test_std[j] = average_of_gradient_metrics(
                gradient_wrt_w_per_sample_test)
            gradient_wrt_x_incorrect_train[j], gradient_wrt_x_incorrect_train_std[j] = average_of_gradient_metrics(
                gradient_wrt_x_per_sample_train)
            gradient_wrt_x_incorrect_test[j], gradient_wrt_x_incorrect_test_std[j] = average_of_gradient_metrics(
                gradient_wrt_x_per_sample_test)

            incorrect_train_samples[j] = gradient_wrt_w_per_sample_train.shape[0]
            incorrect_test_samples[j] = gradient_wrt_w_per_sample_test.shape[0]


            for i in range(7):
                bal_acc_per_class_w_incorrectly_labeled[j, i], acc_per_class_w_incorrectly_labeled[j, i], \
                far_per_class_w_incorrectly_labeled[j, i], prec_per_class_w_incorrectly_labeled[j, i], \
                rcal_per_class_w_incorrectly_labeled[j, i], \
                f1_per_class_w_incorrectly_labeled[j, i] = fit_logistic_regression_model(
                    gradient_wrt_w_per_sample_train[:, i], gradient_wrt_w_per_sample_test[:, i])

                bal_acc_per_class_x_incorrectly_labeled[j, i], acc_per_class_x_incorrectly_labeled[j, i], \
                far_per_class_x_incorrectly_labeled[j, i], prec_per_class_x_incorrectly_labeled[j, i], \
                rcal_per_class_x_incorrectly_labeled[j, i], \
                f1_per_class_x_incorrectly_labeled[j, i] = fit_logistic_regression_model(
                    gradient_wrt_x_per_sample_train[:, i], gradient_wrt_x_per_sample_test[:, i])

            bal_acc_per_class_w_incorrectly_labeled[j, 7], acc_per_class_w_incorrectly_labeled[j, 7], \
            far_per_class_w_incorrectly_labeled[j, 7], prec_per_class_w_incorrectly_labeled[j, 7], \
            rcal_per_class_w_incorrectly_labeled[j, 7], \
            f1_per_class_w_incorrectly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_w_per_sample_train,
                                                                                     gradient_wrt_w_per_sample_test,
                                                                                     single_dimension=False)

            bal_acc_per_class_x_incorrectly_labeled[j, 7], acc_per_class_x_incorrectly_labeled[j, 7], \
            far_per_class_x_incorrectly_labeled[j, 7], prec_per_class_x_incorrectly_labeled[j, 7], \
            rcal_per_class_x_incorrectly_labeled[j, 7], \
            f1_per_class_x_incorrectly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_x_per_sample_train,
                                                                                     gradient_wrt_x_per_sample_test,
                                                                                     single_dimension=False)


    grad_w_correct_train = wigthed_average_for_gradient_metrics(gradient_wrt_w_correct_train, correct_train_samples)
    grad_w_correct_train_std = wigthed_average_for_gradient_metrics(gradient_wrt_w_correct_train_std, correct_train_samples)
    grad_w_correct_test = wigthed_average_for_gradient_metrics(gradient_wrt_w_correct_test, correct_test_samples)
    grad_w_correct_test_std = wigthed_average_for_gradient_metrics(gradient_wrt_w_correct_test_std, correct_test_samples)
    grad_w_incorrect_train = wigthed_average_for_gradient_metrics(gradient_wrt_w_incorrect_train, incorrect_train_samples)
    grad_w_incorrect_train_std = wigthed_average_for_gradient_metrics(gradient_wrt_w_incorrect_train_std, incorrect_train_samples)
    grad_w_incorrect_test = wigthed_average_for_gradient_metrics(gradient_wrt_w_incorrect_test, incorrect_test_samples)
    grad_w_incorrect_test_std = wigthed_average_for_gradient_metrics(gradient_wrt_w_incorrect_test_std, incorrect_test_samples)

    grad_x_correct_train = wigthed_average_for_gradient_metrics(gradient_wrt_x_correct_train, correct_train_samples)
    grad_x_correct_train_std = wigthed_average_for_gradient_metrics(gradient_wrt_x_correct_train_std, correct_train_samples)
    grad_x_correct_test = wigthed_average_for_gradient_metrics(gradient_wrt_x_correct_test, correct_test_samples)
    grad_x_correct_test_std = wigthed_average_for_gradient_metrics(gradient_wrt_x_correct_test_std, correct_test_samples)
    grad_x_incorrect_train = wigthed_average_for_gradient_metrics(gradient_wrt_x_incorrect_train, incorrect_train_samples)
    grad_x_incorrect_train_std = wigthed_average_for_gradient_metrics(gradient_wrt_x_incorrect_train_std, incorrect_train_samples)
    grad_x_incorrect_test = wigthed_average_for_gradient_metrics(gradient_wrt_x_incorrect_test, incorrect_test_samples)
    grad_x_incorrect_test_std = wigthed_average_for_gradient_metrics(gradient_wrt_x_incorrect_test_std, incorrect_test_samples)

    print("\n\nAverage Gradient wrt w: [average standard_deviation]")
    for i in range(7):
        print('Correctly classified (train samples): ', str(np.round(grad_w_correct_train[i], 4)), str(np.round(grad_w_correct_train_std[i], 4)))
        print('Correctly classified (test samples): ', str(np.round(grad_w_correct_test[i], 4)), str(np.round(grad_w_correct_test_std[i])))
        print('Misclassified (train samples): ', str(np.round(grad_w_incorrect_train[i], 4)), str(np.round(grad_w_incorrect_train_std[i], 4)))
        print('Misclassified (test samples): ', str(np.round(grad_w_incorrect_test[i], 4)), str(np.round(grad_w_incorrect_test_std[i], 4)))
        print("\n")

    print("\n\nAverage Gradient wrt x: [average standard_deviation]")
    for i in range(7):
        print('Correctly classified (train samples): ', str(np.round(grad_x_correct_train[i], 4)), str(np.round(grad_x_correct_train_std[i], 4)))
        print('Correctly classified (test samples): ', str(np.round(grad_x_correct_test[i], 4)), str(np.round(grad_x_correct_test_std[i])))
        print('Misclassified (train samples): ', str(np.round(grad_x_incorrect_train[i], 4)), str(np.round(grad_x_incorrect_train_std[i], 4)))
        print('Misclassified (test samples): ', str(np.round(grad_x_incorrect_test[i], 4)), str(np.round(grad_x_incorrect_test_std[i], 4)))
        print("\n")

    bal_acc_w_correct_only, bal_acc_w_correct_only_std = average_of_gradient_metrics(
        bal_acc_per_class_w_correctly_labeled)
    bal_acc_w_incorrect_only, bal_acc_w_incorrect_only_std = average_of_gradient_metrics(
        bal_acc_per_class_w_incorrectly_labeled)
    bal_acc_x_correct_only, bal_acc_x_correct_only_std = average_of_gradient_metrics(
        bal_acc_per_class_x_correctly_labeled)
    bal_acc_x_incorrect_only, bal_acc_x_incorrect_only_std = average_of_gradient_metrics(
        bal_acc_per_class_x_incorrectly_labeled)

    acc_w_correct_only, acc_w_correct_only_std = average_of_gradient_metrics(acc_per_class_w_correctly_labeled)
    acc_w_incorrect_only, acc_w_incorrect_only_std = average_of_gradient_metrics(acc_per_class_w_incorrectly_labeled)
    acc_x_correct_only, acc_x_correct_only_std = average_of_gradient_metrics(acc_per_class_x_correctly_labeled)
    acc_x_incorrect_only, acc_x_incorrect_only_std = average_of_gradient_metrics(acc_per_class_x_incorrectly_labeled)

    far_w_correct_only, far_w_correct_only_std = average_of_gradient_metrics(far_per_class_w_correctly_labeled)
    far_w_incorrect_only, far_w_incorrect_only_std = average_of_gradient_metrics(far_per_class_w_incorrectly_labeled)
    far_x_correct_only, far_x_correct_only_std = average_of_gradient_metrics(far_per_class_x_correctly_labeled)
    far_x_incorrect_only, far_x_incorrect_only_std = average_of_gradient_metrics(far_per_class_x_incorrectly_labeled)

    prec_w_correct_only, prec_w_correct_only_std = average_of_gradient_metrics_of_2d_array(
        prec_per_class_w_correctly_labeled)
    prec_w_incorrect_only, prec_w_incorrect_only_std = average_of_gradient_metrics_of_2d_array(
        prec_per_class_w_incorrectly_labeled)
    prec_x_correct_only, prec_x_correct_only_std = average_of_gradient_metrics_of_2d_array(
        prec_per_class_x_correctly_labeled)
    prec_x_incorrect_only, prec_x_incorrect_only_std = average_of_gradient_metrics_of_2d_array(
        prec_per_class_x_incorrectly_labeled)

    rcal_w_correct_only, rcal_w_correct_only_std = average_of_gradient_metrics_of_2d_array(
        rcal_per_class_w_correctly_labeled)
    rcal_w_incorrect_only, rcal_w_incorrect_only_std = average_of_gradient_metrics_of_2d_array(
        rcal_per_class_w_incorrectly_labeled)
    rcal_x_correct_only, rcal_x_correct_only_std = average_of_gradient_metrics_of_2d_array(
        rcal_per_class_x_correctly_labeled)
    rcal_x_incorrect_only, rcal_x_incorrect_only_std = average_of_gradient_metrics_of_2d_array(
        rcal_per_class_x_incorrectly_labeled)

    f1_w_correct_only, f1_w_correct_only_std = average_of_gradient_metrics_of_2d_array(f1_per_class_w_correctly_labeled)
    f1_w_incorrect_only, f1_w_incorrect_only_std = average_of_gradient_metrics_of_2d_array(
        f1_per_class_w_incorrectly_labeled)
    f1_x_correct_only, f1_x_correct_only_std = average_of_gradient_metrics_of_2d_array(f1_per_class_x_correctly_labeled)
    f1_x_incorrect_only, f1_x_incorrect_only_std = average_of_gradient_metrics_of_2d_array(
        f1_per_class_x_incorrectly_labeled)

    print("\n\n\nAttack bal. accuracy wrt w (separate model per metric):")
    for i in range(7):
        print(str(np.round(bal_acc_w_correct_only[i] * 100, 2)), str(np.round(bal_acc_w_correct_only_std[i] * 100, 2)),
              str(np.round(bal_acc_w_incorrect_only[i] * 100, 2)),
              str(np.round(bal_acc_w_incorrect_only_std[i] * 100, 2)))
    print("\n\n\nAttack bal. accuracy wrt x (separate model per metric):")
    for i in range(7):
        print(str(np.round(bal_acc_x_correct_only[i] * 100, 2)), str(np.round(bal_acc_x_correct_only_std[i] * 100, 2)),
              str(np.round(bal_acc_x_incorrect_only[i] * 100, 2)),
              str(np.round(bal_acc_x_incorrect_only_std[i] * 100, 2)))

    print("\n\n\nAttack accuracy wrt w (separate model per metric):")
    for i in range(7):
        print(str(np.round(acc_w_correct_only[i] * 100, 2)), str(np.round(acc_w_correct_only_std[i] * 100, 2)),
              str(np.round(acc_w_incorrect_only[i] * 100, 2)), str(np.round(acc_w_incorrect_only_std[i] * 100, 2)))
    print("\n\n\nAttack accuracy wrt x (separate model per metric):")
    for i in range(7):
        print(str(np.round(acc_x_correct_only[i] * 100, 2)), str(np.round(acc_x_correct_only_std[i] * 100, 2)),
              str(np.round(acc_x_incorrect_only[i] * 100, 2)), str(np.round(acc_x_incorrect_only_std[i] * 100, 2)))

    print("\n\n\nAttack FAR wrt w (separate model per metric):")
    for i in range(7):
        print(str(np.round(far_w_correct_only[i] * 100, 2)), str(np.round(far_w_correct_only_std[i] * 100, 2)),
              str(np.round(far_w_incorrect_only[i] * 100, 2)), str(np.round(far_w_incorrect_only_std[i] * 100, 2)))
    print("\n\n\nAttack FAR wrt x (separate model per metric):")
    for i in range(7):
        print(str(np.round(far_x_correct_only[i] * 100, 2)), str(np.round(far_x_correct_only_std[i] * 100, 2)),
              str(np.round(far_x_incorrect_only[i] * 100, 2)), str(np.round(far_x_incorrect_only_std[i] * 100, 2)))

    print("\nAttack precision wrt w (separate model per metric):")
    for i in range(7):
        print(str(np.round(prec_w_correct_only[i] * 100, 2)), str(np.round(prec_w_correct_only_std[i] * 100, 2)),
              str(np.round(prec_w_incorrect_only[i] * 100, 2)), str(np.round(prec_w_incorrect_only_std[i] * 100, 2)))
    print("\nAttack precision wrt x (separate model per metric):")
    for i in range(7):
        print(str(np.round(prec_x_correct_only[i] * 100, 2)), str(np.round(prec_x_correct_only_std[i] * 100, 2)),
              str(np.round(prec_x_incorrect_only[i] * 100, 2)), str(np.round(prec_x_incorrect_only_std[i] * 100, 2)))

    print("\nAttack recall wrt w (separate model per metric):")
    for i in range(7):
        print(str(np.round(rcal_w_correct_only[i] * 100, 2)), str(np.round(rcal_w_correct_only_std[i] * 100, 2)),
              str(np.round(rcal_w_incorrect_only[i] * 100, 2)), str(np.round(rcal_w_incorrect_only_std[i] * 100, 2)))
    print("\nAttack recall wrt x (separate model per metric):")
    for i in range(7):
        print(str(np.round(rcal_x_correct_only[i] * 100, 2)), str(np.round(rcal_x_correct_only_std[i] * 100, 2)),
              str(np.round(rcal_x_incorrect_only[i] * 100, 2)), str(np.round(rcal_x_incorrect_only_std[i] * 100, 2)))

    print("\nAttack f1 wrt w (separate model per metric):")
    for i in range(7):
        print(str(np.round(f1_w_correct_only[i] * 100, 2)), str(np.round(f1_w_correct_only_std[i] * 100, 2)),
              str(np.round(f1_w_incorrect_only[i] * 100, 2)), str(np.round(f1_w_incorrect_only_std[i] * 100, 2)))
    print("\nAttack f1 wrt x (separate model per metric):")
    for i in range(7):
        print(str(np.round(f1_x_correct_only[i] * 100, 2)), str(np.round(f1_x_correct_only_std[i] * 100, 2)),
              str(np.round(f1_x_incorrect_only[i] * 100, 2)), str(np.round(f1_x_incorrect_only_std[i] * 100, 2)))



    print("\n\n\nAttack bal. accuracy wrt w (single model for all metrics):")
    print(str(np.round(bal_acc_w_correct_only[7] * 100, 2)), str(np.round(bal_acc_w_correct_only_std[7] * 100, 2)),
          str(np.round(bal_acc_w_incorrect_only[7] * 100, 2)), str(np.round(bal_acc_w_incorrect_only_std[7] * 100, 2)))
    print("Attack accuracy wrt x (single model for all metrics):")
    print(str(np.round(bal_acc_x_correct_only[7] * 100, 2)), str(np.round(bal_acc_x_correct_only_std[7] * 100, 2)),
          str(np.round(bal_acc_x_incorrect_only[7] * 100, 2)), str(np.round(bal_acc_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack accuracy wrt w (single model for all metrics):")
    print(str(np.round(acc_w_correct_only[7] * 100, 2)), str(np.round(acc_w_correct_only_std[7] * 100, 2)),
          str(np.round(acc_w_incorrect_only[7] * 100, 2)), str(np.round(acc_w_incorrect_only_std[7] * 100, 2)))
    print("Attack accuracy wrt x (single model for all metrics):")
    print(str(np.round(acc_x_correct_only[7] * 100, 2)), str(np.round(acc_x_correct_only_std[7] * 100, 2)),
          str(np.round(acc_x_incorrect_only[7] * 100, 2)), str(np.round(acc_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack FAR wrt w (single model for all metrics):")
    print(str(np.round(far_w_correct_only[7] * 100, 2)), str(np.round(far_w_correct_only_std[7] * 100, 2)),
          str(np.round(far_w_incorrect_only[7] * 100, 2)), str(np.round(far_w_incorrect_only_std[7] * 100, 2)))
    print("Attack FAR wrt x (single model for all metrics):")
    print(str(np.round(far_x_correct_only[7] * 100, 2)), str(np.round(far_x_correct_only_std[7] * 100, 2)),
          str(np.round(far_x_incorrect_only[7] * 100, 2)), str(np.round(far_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack precision wrt w (single model for all metrics):")
    print(str(np.round(prec_w_correct_only[7] * 100, 2)), str(np.round(prec_w_correct_only_std[7] * 100, 2)),
          str(np.round(prec_w_incorrect_only[7] * 100, 2)), str(np.round(prec_w_incorrect_only_std[7] * 100, 2)))
    print("Attack precision wrt x (single model for all metrics):")
    print(str(np.round(prec_x_correct_only[7] * 100, 2)), str(np.round(prec_x_correct_only_std[7] * 100, 2)),
          str(np.round(prec_x_incorrect_only[7] * 100, 2)), str(np.round(prec_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack recall wrt w (single model for all metrics):")
    print(str(np.round(rcal_w_correct_only[7] * 100, 2)), str(np.round(rcal_w_correct_only_std[7] * 100, 2)),
          str(np.round(rcal_w_incorrect_only[7] * 100, 2)), str(np.round(rcal_w_incorrect_only_std[7] * 100, 2)))
    print("Attack recall wrt x (single model for all metrics):")
    print(str(np.round(rcal_x_correct_only[7] * 100, 2)), str(np.round(rcal_x_correct_only_std[7] * 100, 2)),
          str(np.round(rcal_x_incorrect_only[7] * 100, 2)), str(np.round(rcal_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack f1 wrt w (single model for all metrics):")
    print(str(np.round(f1_w_correct_only[7] * 100, 2)), str(np.round(f1_w_correct_only_std[7] * 100, 2)),
          str(np.round(f1_w_incorrect_only[7] * 100, 2)), str(np.round(f1_w_incorrect_only_std[7] * 100, 2)))
    print("Attack f1 wrt x (single model for all metrics):")
    print(str(np.round(f1_x_correct_only[7] * 100, 2)), str(np.round(f1_x_correct_only_std[7] * 100, 2)),
          str(np.round(f1_x_incorrect_only[7] * 100, 2)), str(np.round(f1_x_incorrect_only_std[7] * 100, 2)))

