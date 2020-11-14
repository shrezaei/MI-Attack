from __future__ import print_function
import argparse
import os.path
import numpy as np
argparse
from utils import average_over_positive_values, average_over_positive_values_of_2d_array, wigthed_average, false_alarm_rate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score
show_correct_distance = True
show_incorrect_distance = True
index_available_in_the_saved_files = True

#skip attack model training if there is no correctly labeled samples
cor_skip_threshold = 10
#skip attack model training if there is no incorrectly labeled samples
incor_skip_threshold = 10

parser = argparse.ArgumentParser(description='MI attack besed on distance to the boundary.')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-p', '--save_path', type=str, default='saved_distances/cifar10/alexnet', help='Indicate the directory that the computed distances are saved into.')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset
    distance_saved_directory = args.save_path + '/'

    model_save_dir = os.path.join(os.getcwd(), 'saved_models')
    if dataset == "mnist" or dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000
    else:
        print("Unknown dataset!")
        exit()


    distance_correct_train = np.zeros(num_classes) - 1
    distance_correct_train_std = np.zeros(num_classes) - 1
    distance_correct_test = np.zeros(num_classes) - 1
    distance_correct_test_std = np.zeros(num_classes) - 1
    distance_incorrect_train = np.zeros(num_classes) - 1
    distance_incorrect_train_std = np.zeros(num_classes) - 1
    distance_incorrect_test = np.zeros(num_classes) - 1
    distance_incorrect_test_std = np.zeros(num_classes) - 1

    correct_train_samples = np.zeros(num_classes) - 1
    correct_test_samples = np.zeros(num_classes) - 1
    incorrect_train_samples = np.zeros(num_classes) - 1
    incorrect_test_samples = np.zeros(num_classes) - 1


    #To store per-class MI attack accuracy
    bal_acc_per_class_correctly_labeled = np.zeros(num_classes) - 1
    bal_acc_per_class_incorrectly_labeled = np.zeros(num_classes) - 1

    acc_per_class_correctly_labeled = np.zeros(num_classes) - 1
    acc_per_class_incorrectly_labeled = np.zeros(num_classes) - 1

    far_per_class_correctly_labeled = np.zeros(num_classes) - 1
    far_per_class_incorrectly_labeled = np.zeros(num_classes) - 1

    prec_per_class_correctly_labeled = np.zeros((num_classes, 2)) - 1
    prec_per_class_incorrectly_labeled = np.zeros((num_classes, 2)) - 1

    rcal_per_class_correctly_labeled = np.zeros((num_classes, 2)) - 1
    rcal_per_class_incorrectly_labeled = np.zeros((num_classes, 2)) - 1

    f1_per_class_correctly_labeled = np.zeros((num_classes, 2)) - 1
    f1_per_class_incorrectly_labeled = np.zeros((num_classes, 2)) - 1

    def fit_logistic_regression_model(a, b):
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

    for j in range(num_classes):

        if show_correct_distance:
            if index_available_in_the_saved_files:
                train_data_file = distance_saved_directory + 'cor-train-' + str(j) + '.npz'
                test_data_file = distance_saved_directory + 'cor-test-' + str(j) + '.npz'
                if os.path.isfile(train_data_file) and os.path.isfile(test_data_file):
                    distance_per_sample_train = np.nan_to_num(np.load(train_data_file)['arr_0'], posinf=100000,
                                                              neginf=-100000)
                    distance_per_sample_test = np.nan_to_num(np.load(test_data_file)['arr_0'], posinf=100000,
                                                             neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for correctly labeled samples)!")
                    continue
            else:
                train_data_file = distance_saved_directory + 'cor-train-' + str(j) + '.npy'
                test_data_file = distance_saved_directory + 'cor-test-' + str(j) + '.npy'
                if os.path.isfile(train_data_file) and os.path.isfile(test_data_file):
                    distance_per_sample_train = np.nan_to_num(np.load(train_data_file), posinf=100000, neginf=-100000)
                    distance_per_sample_test = np.nan_to_num(np.load(test_data_file), posinf=100000, neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for correctly labeled samples)!")
                    continue

            distance_per_sample_train = distance_per_sample_train[distance_per_sample_train != -1]
            distance_per_sample_test = distance_per_sample_test[distance_per_sample_test != -1]

            distance_correct_train[j], distance_correct_train_std[j] = average_over_positive_values(distance_per_sample_train)
            distance_correct_test[j], distance_correct_test_std[j] = average_over_positive_values(distance_per_sample_test)

            correct_train_samples[j] = distance_per_sample_train.shape[0]
            correct_test_samples[j] = distance_per_sample_test.shape[0]

            #print(correct_train_samples[j], correct_test_samples[j])
            #print(distance_correct_train[j], distance_correct_train_std[j])
            #print(distance_correct_test[j], distance_correct_test_std[j])

            if correct_train_samples[j] < cor_skip_threshold or correct_test_samples[j] < cor_skip_threshold:
                print("Not enough distance sammple is available for class " + str(j) + " (for correctly labeled samples)!")
            else:
                bal_acc_per_class_correctly_labeled[j], acc_per_class_correctly_labeled[j], \
                far_per_class_correctly_labeled[j], prec_per_class_correctly_labeled[j], \
                rcal_per_class_correctly_labeled[j], \
                f1_per_class_correctly_labeled[j] = fit_logistic_regression_model(distance_per_sample_train,
                                                                                  distance_per_sample_test)

        if show_incorrect_distance:
            if index_available_in_the_saved_files:
                train_data_file = distance_saved_directory + 'incor-train-' + str(j) + '.npz'
                test_data_file = distance_saved_directory + 'incor-test-' + str(j) + '.npz'
                if os.path.isfile(train_data_file) and os.path.isfile(test_data_file):
                    distance_per_sample_train = np.nan_to_num(np.load(train_data_file)['arr_0'], posinf=100000,
                                                              neginf=-100000)
                    distance_per_sample_test = np.nan_to_num(np.load(test_data_file)['arr_0'], posinf=100000,
                                                             neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for incorrectly labeled samples)!")
                    continue
            else:
                train_data_file = distance_saved_directory + 'incor-train-' + str(j) + '.npy'
                test_data_file = distance_saved_directory + 'incor-test-' + str(j) + '.npy'
                if os.path.isfile(train_data_file) and os.path.isfile(test_data_file):
                    distance_per_sample_train = np.nan_to_num(np.load(train_data_file), posinf=100000, neginf=-100000)
                    distance_per_sample_test = np.nan_to_num(np.load(test_data_file), posinf=100000, neginf=-100000)
                else:
                    print("No distance file is available for class " + str(j) + " (for incorrectly labeled samples)!")
                    continue

            distance_per_sample_train = distance_per_sample_train[distance_per_sample_train != -1]
            distance_per_sample_test = distance_per_sample_test[distance_per_sample_test != -1]

            distance_incorrect_train[j], distance_incorrect_train_std[j] = average_over_positive_values(distance_per_sample_train)
            distance_incorrect_test[j], distance_incorrect_test_std[j] = average_over_positive_values(distance_per_sample_test)

            incorrect_train_samples[j] = distance_per_sample_train.shape[0]
            incorrect_test_samples[j] = distance_per_sample_test.shape[0]
            #print(incorrect_train_samples[j], incorrect_test_samples[j])
            #print(distance_incorrect_train[j], distance_incorrect_train_std[j])
            #print(distance_incorrect_test[j], distance_incorrect_test_std[j])

            if incorrect_train_samples[j] < incor_skip_threshold or incorrect_test_samples[j] < incor_skip_threshold:
                print("Not enough distance sammple is available for class " + str(j) + " (for incorrectly labeled samples)!")
            else:
                bal_acc_per_class_incorrectly_labeled[j], acc_per_class_incorrectly_labeled[j], \
                far_per_class_incorrectly_labeled[j], prec_per_class_incorrectly_labeled[j], \
                rcal_per_class_incorrectly_labeled[j], \
                f1_per_class_incorrectly_labeled[j] = fit_logistic_regression_model(distance_per_sample_train,
                                                                                    distance_per_sample_test)


    dist_correct_train = wigthed_average(distance_correct_train, correct_train_samples)
    dist_correct_train_std = wigthed_average(distance_correct_train_std, correct_train_samples)
    dist_correct_test = wigthed_average(distance_correct_test, correct_test_samples)
    dist_correct_test_std = wigthed_average(distance_correct_test_std, correct_test_samples)
    dist_incorrect_train = wigthed_average(distance_incorrect_train, incorrect_train_samples)
    dist_incorrect_train_std = wigthed_average(distance_incorrect_train_std, incorrect_train_samples)
    dist_incorrect_test = wigthed_average(distance_incorrect_test, incorrect_test_samples)
    dist_incorrect_test_std = wigthed_average(distance_incorrect_test_std, incorrect_test_samples)

    print("\n\nAverage Distance to Boundary: [average standard_deviation]")
    print('Correctly classified (train samples): ', str(np.round(dist_correct_train, 4)), str(np.round(dist_correct_train_std, 4)))
    print('Correctly classified (test samples): ', str(np.round(dist_correct_test, 4)), str(np.round(dist_correct_test_std, 4)))
    print('Misclassified (train samples): ', str(np.round(dist_incorrect_train, 4)), str(np.round(dist_incorrect_train_std, 4)))
    print('Misclassified (test samples): ', str(np.round(dist_incorrect_test, 4)), str(np.round(dist_incorrect_test_std, 4)))

    bal_acc_correct_only, bal_acc_correct_only_std = average_over_positive_values(bal_acc_per_class_correctly_labeled)
    bal_acc_incorrect_only, bal_acc_incorrect_only_std = average_over_positive_values(bal_acc_per_class_incorrectly_labeled)

    acc_correct_only, acc_correct_only_std = average_over_positive_values(acc_per_class_correctly_labeled)
    acc_incorrect_only, acc_incorrect_only_std = average_over_positive_values(acc_per_class_incorrectly_labeled)

    far_correct_only, far_correct_only_std = average_over_positive_values(far_per_class_correctly_labeled)
    far_incorrect_only, far_incorrect_only_std = average_over_positive_values(far_per_class_incorrectly_labeled)

    prec_correct_only, prec_correct_only_std = average_over_positive_values_of_2d_array(prec_per_class_correctly_labeled)
    prec_incorrect_only, prec_incorrect_only_std = average_over_positive_values_of_2d_array(prec_per_class_incorrectly_labeled)

    rcal_correct_only, rcal_correct_only_std = average_over_positive_values_of_2d_array(rcal_per_class_correctly_labeled)
    rcal_incorrect_only, rcal_incorrect_only_std = average_over_positive_values_of_2d_array(rcal_per_class_incorrectly_labeled)

    f1_correct_only, f1_correct_only_std = average_over_positive_values_of_2d_array(f1_per_class_correctly_labeled)
    f1_incorrect_only, f1_incorrect_only_std = average_over_positive_values_of_2d_array(f1_per_class_incorrectly_labeled)

    print("\n\nAttack accuracy: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(bal_acc_correct_only*100, 2)), str(np.round(bal_acc_correct_only_std*100, 2)))
    print('Misclassified: ', str(np.round(bal_acc_incorrect_only*100, 2)), str(np.round(bal_acc_incorrect_only_std*100, 2)))

    print("\n\nAttack Far: [average standard_deviation]")
    print('Correctly classified: ', str(np.round(far_correct_only*100, 2)), str(np.round(far_correct_only_std*100, 2)))
    print('Misclassified: ', str(np.round(far_incorrect_only*100, 2)), str(np.round(far_incorrect_only_std*100, 2)))

    print("\nAttack precision:")
    print('Correctly classified: ', str(np.round(prec_correct_only*100, 2)), str(np.round(prec_correct_only_std*100, 2)))
    print('Misclassified: ', str(np.round(prec_incorrect_only*100, 2)), str(np.round(prec_incorrect_only_std*100, 2)))

    print("\nAttack recall:")
    print('Correctly classified: ', str(np.round(rcal_correct_only*100, 2)), str(np.round(rcal_correct_only_std*100, 2)))
    print('Misclassified: ', str(np.round(rcal_incorrect_only*100, 2)), str(np.round(rcal_incorrect_only_std*100, 2)))

    print("\nAttack f1:")
    print('Correctly classified: ', str(np.round(f1_correct_only*100, 2)), str(np.round(f1_correct_only_std*100, 2)))
    print('Misclassified: ', str(np.round(f1_incorrect_only*100, 2)), str(np.round(f1_incorrect_only_std*100, 2)))



