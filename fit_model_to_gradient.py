from __future__ import print_function
import os
import os.path
import numpy as np
import pdb
import argparse
from utils import average_of_gradient_metrics_of_2d_array, average_of_gradient_metrics, wigthed_average_for_gradient_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score
show_correct_distance = True
show_incorrect_distance = True


parser = argparse.ArgumentParser(description='MI attack besed on distance to the boundary.')
parser.add_argument('-d', '--dataset', type=str, default='cifar_10', choices=['mnist', 'cifar_10', 'cifar_100', 'cifar_100_resnet', 'cifar_100_densenet', 'imagenet_inceptionv3', 'imagenet_xception'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-m', '--model_path', type=str, default='none', help='Indicate the path to the target model. If you used the train_target_model.py to train the model, leave this field to the default value.')
args = parser.parse_args()



if __name__ == '__main__':
    dataset = args.dataset
    gradient_saved_directory = 'saved_gradients/'

    model_save_dir = os.path.join(os.getcwd(), 'saved_models')
    if dataset == "mnist" or dataset == "cifar_10":
        model_name = model_save_dir + '/' + dataset + '_weights_' + 'final.h5'
        num_classes = 10
    elif dataset == "cifar_100" or dataset == "cifar_100_resnet" or dataset == "cifar_100_densenet":
        model_name = model_save_dir + '/' + dataset + '_weights_' + 'final.h5'
        num_classes = 100
    elif dataset == "imagenet_inceptionv3":
        model_name = model_save_dir + "/imagenet_inceptionV3_v2.hdf5"
        num_classes = 1000
    elif dataset == "imagenet_xception":
        model_name = model_save_dir + "/imagenet_xception_v2.hdf5"
        num_classes = 1000
    else:
        print("Unknown dataset!")
        exit()
    if args.model_path != 'none':
        model_name = args.model_path

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
    acc_per_class_w_correctly_labeled = np.zeros((num_classes, 8)) - 1
    acc_per_class_w_incorrectly_labeled = np.zeros((num_classes, 8)) - 1
    acc_per_class_x_correctly_labeled = np.zeros((num_classes, 8)) - 1
    acc_per_class_x_incorrectly_labeled = np.zeros((num_classes, 8)) - 1

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
        accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)

        return accuracy, precision, recall, f1

    for j in range(num_classes):

        if show_correct_distance:
            train_data_file_w = gradient_saved_directory + model_name.split('/')[-1] + '-cor-w-train-' + str(j) + '.npy'
            test_data_file_w = gradient_saved_directory + model_name.split('/')[-1] + '-cor-w-test-' + str(j) + '.npy'
            train_data_file_x = gradient_saved_directory + model_name.split('/')[-1] + '-cor-x-train-' + str(j) + '.npy'
            test_data_file_x = gradient_saved_directory + model_name.split('/')[-1] + '-cor-x-test-' + str(j) + '.npy'
            if os.path.isfile(train_data_file_w) and os.path.isfile(test_data_file_w) and os.path.isfile(train_data_file_x) and os.path.isfile(test_data_file_x):
                gradient_wrt_w_per_sample_train = np.load(train_data_file_w)
                gradient_wrt_w_per_sample_test = np.load(test_data_file_w)
                gradient_wrt_x_per_sample_train = np.load(train_data_file_x)
                gradient_wrt_x_per_sample_test = np.load(test_data_file_x)
            else:
                print("No gradient file is available for class " + str(j) + " (for correctly labeled samples)!")
                continue

            gradient_wrt_w_correct_train[j], gradient_wrt_w_correct_train_std[j] = average_of_gradient_metrics(gradient_wrt_w_per_sample_train)
            gradient_wrt_w_correct_test[j], gradient_wrt_w_correct_test_std[j] = average_of_gradient_metrics(gradient_wrt_w_per_sample_test)
            gradient_wrt_x_correct_train[j], gradient_wrt_x_correct_train_std[j] = average_of_gradient_metrics(gradient_wrt_x_per_sample_train)
            gradient_wrt_x_correct_test[j], gradient_wrt_x_correct_test_std[j] = average_of_gradient_metrics(gradient_wrt_x_per_sample_test)

            correct_train_samples[j] = gradient_wrt_w_per_sample_train.shape[0]
            correct_test_samples[j] = gradient_wrt_w_per_sample_test.shape[0]

            for i in range(7):
                print(i)
                acc_per_class_w_correctly_labeled[j, i], prec_per_class_w_correctly_labeled[j, i], rcal_per_class_w_correctly_labeled[j, i], \
                f1_per_class_w_correctly_labeled[j, i] = fit_logistic_regression_model(gradient_wrt_w_per_sample_train[:, i], gradient_wrt_w_per_sample_test[:, i])

                acc_per_class_x_correctly_labeled[j, i], prec_per_class_x_correctly_labeled[j, i], rcal_per_class_x_correctly_labeled[j, i], \
                f1_per_class_x_correctly_labeled[j, i] = fit_logistic_regression_model(gradient_wrt_x_per_sample_train[:, i], gradient_wrt_x_per_sample_test[:, i])

            acc_per_class_w_correctly_labeled[j, 7], prec_per_class_w_correctly_labeled[j, 7], rcal_per_class_w_correctly_labeled[j, 7], \
            f1_per_class_w_correctly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_w_per_sample_train, gradient_wrt_w_per_sample_test, single_dimension=False)

            acc_per_class_x_correctly_labeled[j, 7], prec_per_class_x_correctly_labeled[j, 7], rcal_per_class_x_correctly_labeled[j, 7], \
            f1_per_class_x_correctly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_x_per_sample_train, gradient_wrt_x_per_sample_test, single_dimension=False)

        if show_incorrect_distance:
            train_data_file_w = gradient_saved_directory + model_name.split('/')[-1] + '-incor-w-train-' + str(j) + '.npy'
            test_data_file_w = gradient_saved_directory + model_name.split('/')[-1] + '-incor-w-test-' + str(j) + '.npy'
            train_data_file_x = gradient_saved_directory + model_name.split('/')[-1] + '-incor-x-train-' + str(j) + '.npy'
            test_data_file_x = gradient_saved_directory + model_name.split('/')[-1] + '-incor-x-test-' + str(j) + '.npy'
            if os.path.isfile(train_data_file_w) and os.path.isfile(test_data_file_w) and os.path.isfile(train_data_file_x) and os.path.isfile(test_data_file_x):
                gradient_wrt_w_per_sample_train = np.load(train_data_file_w)
                gradient_wrt_w_per_sample_test = np.load(test_data_file_w)
                gradient_wrt_x_per_sample_train = np.load(train_data_file_x)
                gradient_wrt_x_per_sample_test = np.load(test_data_file_x)
            else:
                print("No distance file is available for class " + str(j) + " (for incorrectly labeled samples)!")
                continue

            gradient_wrt_w_incorrect_train[j], gradient_wrt_w_incorrect_train_std[j] = average_of_gradient_metrics(gradient_wrt_w_per_sample_train)
            gradient_wrt_w_incorrect_test[j], gradient_wrt_w_incorrect_test_std[j] = average_of_gradient_metrics(gradient_wrt_w_per_sample_test)
            gradient_wrt_x_incorrect_train[j], gradient_wrt_x_incorrect_train_std[j] = average_of_gradient_metrics(gradient_wrt_x_per_sample_train)
            gradient_wrt_x_incorrect_test[j], gradient_wrt_x_incorrect_test_std[j] = average_of_gradient_metrics(gradient_wrt_x_per_sample_test)

            incorrect_train_samples[j] = gradient_wrt_w_per_sample_train.shape[0]
            incorrect_test_samples[j] = gradient_wrt_w_per_sample_test.shape[0]

            for i in range(7):
                acc_per_class_w_incorrectly_labeled[j, i], prec_per_class_w_incorrectly_labeled[j, i], rcal_per_class_w_incorrectly_labeled[j, i], \
                f1_per_class_w_incorrectly_labeled[j, i] = fit_logistic_regression_model(gradient_wrt_w_per_sample_train[:, i], gradient_wrt_w_per_sample_test[:, i])

                acc_per_class_x_incorrectly_labeled[j, i], prec_per_class_x_incorrectly_labeled[j, i], rcal_per_class_x_incorrectly_labeled[j, i], \
                f1_per_class_x_incorrectly_labeled[j, i] = fit_logistic_regression_model(gradient_wrt_x_per_sample_train[:, i], gradient_wrt_x_per_sample_test[:, i])

            acc_per_class_w_incorrectly_labeled[j, 7], prec_per_class_w_incorrectly_labeled[j, 7], rcal_per_class_w_incorrectly_labeled[j, 7], \
            f1_per_class_w_incorrectly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_w_per_sample_train, gradient_wrt_w_per_sample_test, single_dimension=False)

            acc_per_class_x_incorrectly_labeled[j, 7], prec_per_class_x_incorrectly_labeled[j, 7], rcal_per_class_x_incorrectly_labeled[j, 7], \
            f1_per_class_x_incorrectly_labeled[j, 7] = fit_logistic_regression_model(gradient_wrt_x_per_sample_train, gradient_wrt_x_per_sample_test, single_dimension=False)


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

    acc_w_correct_only, acc_w_correct_only_std = average_of_gradient_metrics(acc_per_class_w_correctly_labeled)
    acc_w_incorrect_only, acc_w_incorrect_only_std = average_of_gradient_metrics(acc_per_class_w_incorrectly_labeled)
    acc_x_correct_only, acc_x_correct_only_std = average_of_gradient_metrics(acc_per_class_x_correctly_labeled)
    acc_x_incorrect_only, acc_x_incorrect_only_std = average_of_gradient_metrics(acc_per_class_x_incorrectly_labeled)

    prec_w_correct_only, prec_w_correct_only_std = average_of_gradient_metrics_of_2d_array(prec_per_class_w_correctly_labeled)
    prec_w_incorrect_only, prec_w_incorrect_only_std = average_of_gradient_metrics_of_2d_array(prec_per_class_w_incorrectly_labeled)
    prec_x_correct_only, prec_x_correct_only_std = average_of_gradient_metrics_of_2d_array(prec_per_class_x_correctly_labeled)
    prec_x_incorrect_only, prec_x_incorrect_only_std = average_of_gradient_metrics_of_2d_array(prec_per_class_x_incorrectly_labeled)

    rcal_w_correct_only, rcal_w_correct_only_std = average_of_gradient_metrics_of_2d_array(rcal_per_class_w_correctly_labeled)
    rcal_w_incorrect_only, rcal_w_incorrect_only_std = average_of_gradient_metrics_of_2d_array(rcal_per_class_w_incorrectly_labeled)
    rcal_x_correct_only, rcal_x_correct_only_std = average_of_gradient_metrics_of_2d_array(rcal_per_class_x_correctly_labeled)
    rcal_x_incorrect_only, rcal_x_incorrect_only_std = average_of_gradient_metrics_of_2d_array(rcal_per_class_x_incorrectly_labeled)

    f1_w_correct_only, f1_w_correct_only_std = average_of_gradient_metrics_of_2d_array(f1_per_class_w_correctly_labeled)
    f1_w_incorrect_only, f1_w_incorrect_only_std = average_of_gradient_metrics_of_2d_array(f1_per_class_w_incorrectly_labeled)
    f1_x_correct_only, f1_x_correct_only_std = average_of_gradient_metrics_of_2d_array(f1_per_class_x_correctly_labeled)
    f1_x_incorrect_only, f1_x_incorrect_only_std = average_of_gradient_metrics_of_2d_array(f1_per_class_x_incorrectly_labeled)

    print("\n\n\nAttack accuracy wrt w: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(acc_w_correct_only[7] * 100, 2)), str(np.round(acc_w_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(acc_w_incorrect_only[7] * 100, 2)), str(np.round(acc_w_incorrect_only_std[7] * 100, 2)))
    print("Attack accuracy wrt x: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(acc_x_correct_only[7] * 100, 2)), str(np.round(acc_x_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(acc_x_incorrect_only[7] * 100, 2)), str(np.round(acc_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack precision wrt w: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(prec_w_correct_only[7] * 100, 2)), str(np.round(prec_w_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(prec_w_incorrect_only[7] * 100, 2)), str(np.round(prec_w_incorrect_only_std[7] * 100, 2)))
    print("Attack precision wrt x: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(prec_x_correct_only[7] * 100, 2)), str(np.round(prec_x_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(prec_x_incorrect_only[7] * 100, 2)), str(np.round(prec_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack recall wrt w: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(rcal_w_correct_only[7] * 100, 2)), str(np.round(rcal_w_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(rcal_w_incorrect_only[7] * 100, 2)), str(np.round(rcal_w_incorrect_only_std[7] * 100, 2)))
    print("Attack recall wrt x: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(rcal_x_correct_only[7] * 100, 2)), str(np.round(rcal_x_correct_only_std[7] * 100, 2)), str(np.round(rcal_x_incorrect_only[7] * 100, 2)), str(np.round(rcal_x_incorrect_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(rcal_x_incorrect_only[7] * 100, 2)), str(np.round(rcal_x_incorrect_only_std[7] * 100, 2)))

    print("\nAttack f1 wrt w: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(f1_w_correct_only[7] * 100, 2)), str(np.round(f1_w_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(f1_w_incorrect_only[7] * 100, 2)), str(np.round(f1_w_incorrect_only_std[7] * 100, 2)))
    print("Attack f1 wrt x: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(f1_x_correct_only[7] * 100, 2)), str(np.round(f1_x_correct_only_std[7] * 100, 2)))
    print('Misclassified: ', str(np.round(f1_x_incorrect_only[7] * 100, 2)), str(np.round(f1_x_incorrect_only_std[7] * 100, 2)))
