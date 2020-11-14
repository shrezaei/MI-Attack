from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
from matplotlib import rcParams
from utils import average_over_positive_values, average_over_positive_values_of_2d_array, false_alarm_rate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

rcParams.update({'font.size': 16})
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

balanceness_ratio = 5
apply_sampling_to_test_as_well = False
show_MI_attack = True
show_blind_attack = True

def normalize_data(data, means, stds):
    data /= 255
    data[:, :, :, 0] -= means[0]
    data[:, :, :, 0] /= stds[0]
    data[:, :, :, 1] -= means[1]
    data[:, :, :, 1] /= stds[1]
    data[:, :, :, 2] -= means[2]
    data[:, :, :, 2] /= stds[2]
    return data

def conf_based_attack(dataset, attack_classifier, sampling, what_portion_of_samples_attacker_knows, save_confidence_histogram, report_separated_performance, num_classes, num_targeted_classes, model_name, verbose):
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

    model = tf.keras.models.load_model(model_name)

    train_stat = model.evaluate(x_train, y_train, verbose=0)
    test_stat = model.evaluate(x_test, y_test, verbose=0)

    acc_train = train_stat[1]
    acc_test = test_stat[1]

    print(acc_train, acc_test)

    confidence_train = model.predict(x_train)
    confidence_test = model.predict(x_test)
    labels_train_by_model = np.argmax(confidence_train, axis=1)
    labels_test_by_model = np.argmax(confidence_test, axis=1)
    labels_train = np.argmax(y_train, axis=1)
    labels_test = np.argmax(y_test, axis=1)

    temp_indexer = np.arange(confidence_train.shape[0])
    temp_all_conf_array = confidence_train[temp_indexer, labels_train]
    conf_train = np.average(temp_all_conf_array)
    conf_train_std = np.std(temp_all_conf_array)

    correctly_classified_indexes_train = labels_train_by_model == labels_train
    temp_correct_conf_array = confidence_train[temp_indexer[correctly_classified_indexes_train], labels_train[correctly_classified_indexes_train]]
    conf_train_correct_only = np.average(temp_correct_conf_array)
    conf_train_correct_only_std = np.std(temp_correct_conf_array)

    incorrectly_classified_indexes_train = labels_train_by_model != labels_train
    temp_incorrect_conf_array = confidence_train[temp_indexer[incorrectly_classified_indexes_train], labels_train_by_model[incorrectly_classified_indexes_train]]
    conf_train_incorrect_only = np.average(temp_incorrect_conf_array)
    conf_train_incorrect_only_std = np.std(temp_incorrect_conf_array)

    # Compute average confidence for test set
    temp_indexer = np.arange(confidence_test.shape[0])
    temp_all_conf_array = confidence_test[temp_indexer, labels_test]
    conf_test = np.average(temp_all_conf_array)
    conf_test_std = np.std(temp_all_conf_array)

    correctly_classified_indexes_test = labels_test_by_model == labels_test
    temp_correct_conf_array = confidence_test[temp_indexer[correctly_classified_indexes_test], labels_test[correctly_classified_indexes_test]]
    conf_test_correct_only = np.average(temp_correct_conf_array)
    conf_test_correct_only_std = np.std(temp_correct_conf_array)

    incorrectly_classified_indexes_test = labels_test_by_model != labels_test
    temp_incorrect_conf_array = confidence_test[temp_indexer[incorrectly_classified_indexes_test], labels_test_by_model[incorrectly_classified_indexes_test]]
    conf_test_incorrect_only = np.average(temp_incorrect_conf_array)
    conf_test_incorrect_only_std = np.std(temp_incorrect_conf_array)

    #To store per-class MI attack accuracy
    MI_attack_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_acc_per_class = np.zeros(num_targeted_classes) - 1  # accuracy
    MI_attack_acc_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_acc_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_far_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_far_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_far_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_prec_per_class = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_prec_per_class_correctly_labeled = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_prec_per_class_incorrectly_labeled = np.zeros((num_targeted_classes, 2)) - 1

    MI_attack_rcal_per_class = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_correctly_labeled = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_incorrectly_labeled = np.zeros((num_targeted_classes, 2)) - 1

    MI_attack_f1_per_class = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_correctly_labeled = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_incorrectly_labeled = np.zeros((num_targeted_classes, 2)) - 1

    # To store per-class MI blind attack accuracy: return 1 if classifier classify correctly, otherwise 0
    MI_attack_blind_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_blind_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_blind_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_blind_acc_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_blind_acc_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_blind_acc_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_blind_far_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_blind_far_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_blind_far_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_blind_prec_per_class = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_blind_prec_per_class_correctly_labeled = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_blind_prec_per_class_incorrectly_labeled = np.zeros((num_targeted_classes, 2)) - 1

    MI_attack_blind_rcal_per_class = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_blind_rcal_per_class_correctly_labeled = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_blind_rcal_per_class_incorrectly_labeled = np.zeros((num_targeted_classes, 2)) - 1

    MI_attack_blind_f1_per_class = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_blind_f1_per_class_correctly_labeled = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_blind_f1_per_class_incorrectly_labeled = np.zeros((num_targeted_classes, 2)) - 1

    for j in range(num_targeted_classes):
        #Prepare the data for training and testing attack models (for all data and also correctly labeled samples)
        class_yes_x = confidence_train[tuple([labels_train == j])]
        class_no_x = confidence_test[tuple([labels_test == j])]

        if class_yes_x.shape[0] < 15 or class_no_x.shape[0] < 15:
            print("Class " + str(j) + " doesn't have enough sample for training an attack model!")
            continue

        class_yes_x_correctly_labeled = correctly_classified_indexes_train[tuple([labels_train == j])]
        class_no_x_correctly_labeled = correctly_classified_indexes_test[tuple([labels_test == j])]

        class_yes_x_incorrectly_labeled = incorrectly_classified_indexes_train[tuple([labels_train == j])]
        class_no_x_incorrectly_labeled = incorrectly_classified_indexes_test[tuple([labels_test == j])]

        if save_confidence_histogram:
            temp = class_yes_x[class_yes_x_correctly_labeled]
            temp2 = class_no_x[class_no_x_correctly_labeled]
            temp = np.average(temp, axis=0)
            temp2 = np.average(temp2, axis=0)
            plt.style.use('seaborn-deep')
            plt.plot(np.arange(num_classes), temp, 'bx', label="Train samples")
            plt.plot(np.arange(num_classes), temp2, 'go', label="Test samples")
            plt.legend()
            plt.xlabel("Class Number")
            plt.ylabel("Average Confidence")
            plt.savefig('figures/conf histogram/' + dataset + '/correct-' + str(j) + '.eps', bbox_inches='tight')
            plt.close()

            temp = class_yes_x[class_yes_x_incorrectly_labeled]
            temp2 = class_no_x[class_no_x_incorrectly_labeled]
            temp = np.average(temp, axis=0)
            temp2 = np.average(temp2, axis=0)
            plt.style.use('seaborn-deep')
            plt.plot(np.arange(num_classes), temp, 'bx', label="Train samples")
            plt.plot(np.arange(num_classes), temp2, 'go', label="Test samples")
            plt.legend()
            plt.xlabel("Class Number")
            plt.ylabel("Average Confidence")
            plt.savefig('figures/conf histogram/' + dataset + '/misclassified-' + str(j) + '.eps', bbox_inches='tight')
            plt.close()

            temp = class_yes_x[class_yes_x_correctly_labeled]
            temp2 = class_no_x[class_no_x_correctly_labeled]
            bins = np.arange(101) / 100
            plt.style.use('seaborn-deep')
            n, bins, patches = plt.hist([temp[:, j], temp2[:, j]], bins, normed=1, alpha=1, label=['Train samples', 'Test samples'])
            plt.xlabel('Model Confidence')
            plt.ylabel('Probability (%)')
            plt.legend(loc='upper left')
            plt.savefig('figures/conf histogram/' + dataset + '/' + str(j) + '.eps', bbox_inches='tight')
            plt.close()


        class_yes_size = int(class_yes_x.shape[0] * what_portion_of_samples_attacker_knows)
        class_yes_x_train = class_yes_x[:class_yes_size]
        class_yes_y_train = np.ones(class_yes_x_train.shape[0])
        class_yes_x_test = class_yes_x[class_yes_size:]
        class_yes_y_test = np.ones(class_yes_x_test.shape[0])
        class_yes_x_correctly_labeled = class_yes_x_correctly_labeled[class_yes_size:]
        class_yes_x_incorrectly_labeled = class_yes_x_incorrectly_labeled[class_yes_size:]

        class_no_size = int(class_no_x.shape[0] * what_portion_of_samples_attacker_knows)
        class_no_x_train = class_no_x[:class_no_size]
        class_no_y_train = np.zeros(class_no_x_train.shape[0])
        class_no_x_test = class_no_x[class_no_size:]
        class_no_y_test = np.zeros(class_no_x_test.shape[0])
        class_no_x_correctly_labeled = class_no_x_correctly_labeled[class_no_size:]
        class_no_x_incorrectly_labeled = class_no_x_incorrectly_labeled[class_no_size:]

        y_size = class_yes_x_train.shape[0]
        n_size = class_no_x_train.shape[0]
        if sampling == "undersampling":
            if y_size > n_size:
                class_yes_x_train = class_yes_x_train[:n_size]
                class_yes_y_train = class_yes_y_train[:n_size]
            else:
                class_no_x_train = class_no_x_train[:y_size]
                class_no_y_train = class_no_y_train[:y_size]
        elif sampling == "oversampling":
            if y_size > n_size:
                class_no_x_train = np.tile(class_no_x_train, (int(y_size / n_size), 1))
                class_no_y_train = np.zeros(class_no_x_train.shape[0])
            else:
                class_yes_x_train = np.tile(class_yes_x_train, (int(n_size / y_size), 1))
                class_yes_y_train = np.ones(class_yes_x_train.shape[0])
        elif sampling == "revert_balanceness":
            y_final_size = int(n_size / balanceness_ratio)
            class_yes_x_train = class_yes_x_train[:y_final_size]
            class_yes_y_train = class_yes_y_train[:y_final_size]

        if apply_sampling_to_test_as_well:
            y_size = class_yes_x_test.shape[0]
            n_size = class_no_x_test.shape[0]
            if sampling == "undersampling":
                if y_size > n_size:
                    class_yes_x_test = class_yes_x_test[:n_size]
                    class_yes_y_test = class_yes_y_test[:n_size]
                else:
                    class_no_x_test = class_no_x_test[:y_size]
                    class_no_y_test = class_no_y_test[:y_size]
            elif sampling == "oversampling":
                if y_size > n_size:
                    class_no_x_test = np.tile(class_no_x_test, (int(y_size / n_size), 1))
                    class_no_y_test = np.zeros(class_no_x_test.shape[0])
                else:
                    class_yes_x_test = np.tile(class_yes_x_test, (int(n_size / y_size), 1))
                    class_yes_y_test = np.ones(class_yes_x_test.shape[0])
            elif sampling == "revert_balanceness":
                y_final_size = int(n_size / balanceness_ratio)
                class_yes_x_test = class_yes_x_test[:y_final_size]
                class_yes_y_test = class_yes_y_test[:y_final_size]

        print('MI attack on class ', j)
        MI_x_train = np.concatenate((class_yes_x_train, class_no_x_train), axis=0)
        MI_y_train = np.concatenate((class_yes_y_train, class_no_y_train), axis=0)
        MI_x_test = np.concatenate((class_yes_x_test, class_no_x_test), axis=0)
        MI_y_test = np.concatenate((class_yes_y_test, class_no_y_test), axis=0)
        MI_correctly_labeled_indexes = np.concatenate((class_yes_x_correctly_labeled, class_no_x_correctly_labeled), axis=0)
        MI_incorrectly_labeled_indexes = np.concatenate((class_yes_x_incorrectly_labeled, class_no_x_incorrectly_labeled), axis=0)


        # #preparing data to train an attack model for incorrectly labeled samples
        # if show_MI_attack_separate_result:
        #     cor_class_yes_x = confidence_train[correctly_classified_indexes_train]
        #     cor_class_no_x = confidence_test[correctly_classified_indexes_test]
        #     cor_class_yes_x = cor_class_yes_x[np.argmax(cor_class_yes_x, axis=1) == j]
        #     cor_class_no_x = cor_class_no_x[np.argmax(cor_class_no_x, axis=1) == j]
        #
        #     if cor_class_yes_x.shape[0] < 15 or cor_class_no_x.shape[0] < 15:
        #         print("Class " + str(j) + " doesn't have enough sample for training an attack model!")
        #         continue
        #
        #     cor_class_yes_size = int(cor_class_yes_x.shape[0] * what_portion_of_samples_attacker_knows)
        #     cor_class_no_size = int(cor_class_no_x.shape[0] * what_portion_of_samples_attacker_knows)
        #
        #     cor_class_yes_x_train = cor_class_yes_x[:cor_class_yes_size]
        #     cor_class_yes_y_train = np.ones(cor_class_yes_x_train.shape[0])
        #     cor_class_yes_x_test = cor_class_yes_x[cor_class_yes_size:]
        #     cor_class_yes_y_test = np.ones(cor_class_yes_x_test.shape[0])
        #
        #     cor_class_no_x_train = cor_class_no_x[:cor_class_no_size]
        #     cor_class_no_y_train = np.zeros(cor_class_no_x_train.shape[0])
        #     cor_class_no_x_test = cor_class_no_x[cor_class_no_size:]
        #     cor_class_no_y_test = np.zeros(cor_class_no_x_test.shape[0])
        #
        #     y_size = cor_class_yes_x_train.shape[0]
        #     n_size = cor_class_no_x_train.shape[0]
        #     if sampling == "undersampling":
        #         if y_size > n_size:
        #             cor_class_yes_x_train = cor_class_yes_x_train[:n_size]
        #             cor_class_yes_y_train = cor_class_yes_y_train[:n_size]
        #         else:
        #             cor_class_no_x_train = cor_class_no_x_train[:y_size]
        #             cor_class_no_y_train = cor_class_no_y_train[:y_size]
        #     elif sampling == "oversampling":
        #         if y_size > n_size:
        #             cor_class_no_x_train = np.tile(cor_class_no_x_train, (int(y_size / n_size), 1))
        #             cor_class_no_y_train = np.zeros(cor_class_no_x_train.shape[0])
        #         else:
        #             cor_class_yes_x_train = np.tile(cor_class_yes_x_train, (int(n_size / y_size), 1))
        #             cor_class_yes_y_train = np.ones(cor_class_yes_x_train.shape[0])
        #     elif sampling == "revert_balanceness":
        #         y_final_size = int(n_size / balanceness_ratio)
        #         cor_class_yes_x_train = cor_class_yes_x_train[:y_final_size]
        #         cor_class_yes_y_train = cor_class_yes_y_train[:y_final_size]
        #
        #     cor_MI_x_train = np.concatenate((cor_class_yes_x_train, cor_class_no_x_train), axis=0)
        #     cor_MI_y_train = np.concatenate((cor_class_yes_y_train, cor_class_no_y_train), axis=0)
        #     cor_MI_x_test = np.concatenate((cor_class_yes_x_test, cor_class_no_x_test), axis=0)
        #     cor_MI_y_test = np.concatenate((cor_class_yes_y_test, cor_class_no_y_test), axis=0)
        #
        #     if apply_sampling_to_test_as_well:
        #         y_size = cor_class_yes_x_test.shape[0]
        #         n_size = cor_class_no_x_test.shape[0]
        #         if sampling == "undersampling":
        #             if y_size > n_size:
        #                 cor_class_yes_x_test = cor_class_yes_x_test[:n_size]
        #                 cor_class_yes_y_test = cor_class_yes_y_test[:n_size]
        #             else:
        #                 cor_class_no_x_test = cor_class_no_x_test[:y_size]
        #                 cor_class_no_y_test = cor_class_no_y_test[:y_size]
        #         elif sampling == "oversampling":
        #             if y_size > n_size:
        #                 cor_class_no_x_test = np.tile(cor_class_no_x_test, (int(y_size / n_size), 1))
        #                 cor_class_no_y_test = np.zeros(cor_class_no_x_test.shape[0])
        #             else:
        #                 cor_class_yes_x_test = np.tile(cor_class_yes_x_test, (int(n_size / y_size), 1))
        #                 cor_class_yes_y_test = np.ones(cor_class_yes_x_test.shape[0])
        #         elif sampling == "revert_balanceness":
        #             y_final_size = int(n_size / balanceness_ratio)
        #             cor_class_yes_x_test = cor_class_yes_x_test[:y_final_size]
        #             cor_class_yes_y_test = cor_class_yes_y_test[:y_final_size]

        if show_MI_attack:
            if attack_classifier == "NN":
                # Use NN classifier to launch Membership Inference attack (All data + correctly labeled)
                attack_model = Sequential()
                attack_model.add(Dense(128, input_dim=num_classes, activation='relu'))
                attack_model.add(Dense(64, activation='relu'))
                attack_model.add(Dense(1, activation='sigmoid'))
                attack_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
                attack_model.fit(MI_x_train, MI_y_train, validation_data=(MI_x_test, MI_y_test), epochs=30, batch_size=32, verbose=False, shuffle=True)

            elif attack_classifier == "RF":
                n_est = [500, 800, 1500, 2500, 5000]
                max_f = ['auto', 'sqrt']
                max_depth = [20, 30, 40, 50]
                max_depth.append(None)
                min_samples_s = [2, 5, 10, 15, 20]
                min_samples_l = [1, 2, 5, 10, 15]
                grid_param = {'n_estimators': n_est,
                              'max_features': max_f,
                              'max_depth': max_depth,
                              'min_samples_split': min_samples_s,
                              'min_samples_leaf': min_samples_l}
                RFR = RandomForestClassifier(random_state=1)
                if verbose:
                    RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=100, cv=2, verbose=1, random_state=42, n_jobs=-1)
                else:
                    RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=100, cv=2, verbose=0, random_state=42, n_jobs=-1)
                RFR_random.fit(MI_x_train, MI_y_train)
                if verbose:
                    print(RFR_random.best_params_)
                attack_model = RFR_random.best_estimator_


            elif attack_classifier == "XGBoost":
                temp_model = XGBClassifier()
                param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100], min_child_weight=[1, 5, 10, 15], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], max_depth=[3, 6, 9, 12])
                # param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100, 500, 1000])
                cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1)
                # grid = GridSearchCV(estimator=temp_model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
                grid = RandomizedSearchCV(estimator=temp_model, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
                grid_result = grid.fit(MI_x_train, MI_y_train)
                attack_model = grid_result.best_estimator_
                if verbose:
                    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

            # MI attack accuracy on all data
            if attack_classifier == "NN":
                y_pred = attack_model.predict_classes(MI_x_test)
            else:
                y_pred = attack_model.predict(MI_x_test)
            MI_attack_per_class[j] = balanced_accuracy_score(MI_y_test, y_pred)
            MI_attack_acc_per_class[j] = accuracy_score(MI_y_test, y_pred)
            MI_attack_far_per_class[j] = false_alarm_rate(MI_y_test, y_pred)
            MI_attack_prec_per_class[j] = precision_score(MI_y_test, y_pred, average=None)
            MI_attack_rcal_per_class[j] = recall_score(MI_y_test, y_pred, average=None)
            MI_attack_f1_per_class[j] = f1_score(MI_y_test, y_pred, average=None)

            if report_separated_performance:
                # MI attack accuracy on correctly labeled
                if np.sum(MI_correctly_labeled_indexes) > 0:
                    temp_x = MI_x_test[MI_correctly_labeled_indexes]
                    temp_y = MI_y_test[MI_correctly_labeled_indexes]
                    if attack_classifier == "NN":
                        y_pred = attack_model.predict_classes(temp_x)
                    else:
                        y_pred = attack_model.predict(temp_x)
                    MI_attack_per_class_correctly_labeled[j] = balanced_accuracy_score(temp_y, y_pred)
                    MI_attack_acc_per_class_correctly_labeled[j] = accuracy_score(temp_y, y_pred)
                    MI_attack_far_per_class_correctly_labeled[j] = false_alarm_rate(temp_y, y_pred)
                    MI_attack_prec_per_class_correctly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                    MI_attack_rcal_per_class_correctly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                    MI_attack_f1_per_class_correctly_labeled[j] = f1_score(temp_y, y_pred, average=None)

                # MI attack accuracy on incorrectly labeled
                if np.sum(MI_incorrectly_labeled_indexes) > 0:
                    temp_x = MI_x_test[MI_incorrectly_labeled_indexes]
                    temp_y = MI_y_test[MI_incorrectly_labeled_indexes]
                    if attack_classifier == "NN":
                        y_pred = attack_model.predict_classes(temp_x)
                    else:
                        y_pred = attack_model.predict(temp_x)
                    MI_attack_per_class_incorrectly_labeled[j] = balanced_accuracy_score(temp_y, y_pred)
                    MI_attack_acc_per_class_incorrectly_labeled[j] = accuracy_score(temp_y, y_pred)
                    MI_attack_far_per_class_incorrectly_labeled[j] = false_alarm_rate(temp_y, y_pred)
                    MI_attack_prec_per_class_incorrectly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                    MI_attack_rcal_per_class_incorrectly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                    MI_attack_f1_per_class_incorrectly_labeled[j] = f1_score(temp_y, y_pred, average=None)

            if verbose:
                print('\nMI Attack Bal. Acc (all data):', MI_attack_per_class[j], MI_x_test.shape[0], np.sum([MI_y_train == 0]), np.sum([MI_y_train == 1]), np.sum([MI_y_test == 0]), np.sum([MI_y_test == 1]))
                print('MI Attack(FAR):', MI_attack_far_per_class[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class[j])
                print('MI Attack(F1):', MI_attack_f1_per_class[j])

                print('\nMI Attack Bal. Acc (correctly classified):', MI_attack_per_class_correctly_labeled[j],
                      np.sum(MI_correctly_labeled_indexes), np.sum(MI_y_test[MI_correctly_labeled_indexes] == 0),
                      np.sum(MI_y_test[MI_correctly_labeled_indexes] == 1))
                print('MI Attack(FAR):', MI_attack_far_per_class_correctly_labeled[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_correctly_labeled[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_correctly_labeled[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_correctly_labeled[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_correctly_labeled[j])
                print('\nMI Attack Bal. Acc (Misclassified):', MI_attack_per_class_incorrectly_labeled[j],
                      np.sum(MI_incorrectly_labeled_indexes), np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 0),
                      np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 1))
                print('MI Attack(FAR):', MI_attack_far_per_class_incorrectly_labeled[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_incorrectly_labeled[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_incorrectly_labeled[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_incorrectly_labeled[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_incorrectly_labeled[j])

        if show_blind_attack:
            # MI_x_train_blind = MI_x_train[:, j]     #To be fare, I just use the test test, to compare with other attack, so I comment it
            MI_x_test_blind = np.argmax(MI_x_test, axis=1)
            MI_predicted_y_test_blind = [1 if l==j else 0 for l in MI_x_test_blind]
            MI_predicted_y_test_blind = np.array(MI_predicted_y_test_blind)

            y_pred = MI_predicted_y_test_blind
            if y_pred.shape[0] > 0:
                y_pred = MI_predicted_y_test_blind
                MI_attack_blind_per_class[j] = balanced_accuracy_score(MI_y_test, y_pred)
                MI_attack_blind_acc_per_class[j] = accuracy_score(MI_y_test, y_pred)
                MI_attack_blind_far_per_class[j] = false_alarm_rate(MI_y_test, y_pred)
                MI_attack_blind_prec_per_class[j] = precision_score(MI_y_test, y_pred, average=None)
                MI_attack_blind_rcal_per_class[j] = recall_score(MI_y_test, y_pred, average=None)
                MI_attack_blind_f1_per_class[j] = f1_score(MI_y_test, y_pred, average=None)

                print('\nMI Naive Attack (all):', MI_attack_blind_per_class[j], MI_x_test.shape[0],
                      np.sum([MI_y_train == 0]), np.sum([MI_y_train == 1]), np.sum([MI_y_test == 0]),
                      np.sum([MI_y_test == 1]))
                print('MI Naive Attack(FAR):', MI_attack_blind_far_per_class[j])
                print('MI Naive Attack(Acc-unbalanced):', MI_attack_blind_acc_per_class[j])
                print('MI Naive Attack(Prec):', MI_attack_blind_prec_per_class[j])
                print('MI Naive Attack(Rec):', MI_attack_blind_rcal_per_class[j])
                print('MI Naive Attack(F1):', MI_attack_blind_f1_per_class[j])

            if report_separated_performance:
                # MI dumpattack accuracy on correctly labeled
                if np.sum(MI_correctly_labeled_indexes) > 0:
                    temp_y = MI_y_test[MI_correctly_labeled_indexes]
                    y_pred = MI_predicted_y_test_blind[MI_correctly_labeled_indexes]
                    if y_pred.shape[0] > 0:
                        MI_attack_blind_per_class_correctly_labeled[j] = balanced_accuracy_score(temp_y, y_pred)
                        MI_attack_blind_acc_per_class_correctly_labeled[j] = accuracy_score(temp_y, y_pred)
                        MI_attack_blind_far_per_class_correctly_labeled[j] = false_alarm_rate(temp_y, y_pred)
                        MI_attack_blind_prec_per_class_correctly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                        MI_attack_blind_rcal_per_class_correctly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                        MI_attack_blind_f1_per_class_correctly_labeled[j] = f1_score(temp_y, y_pred, average=None)

                # MI dump attack accuracy on incorrectly labeled
                if np.sum(MI_incorrectly_labeled_indexes) > 0:
                    temp_y = MI_y_test[MI_incorrectly_labeled_indexes]
                    y_pred = MI_predicted_y_test_blind[MI_incorrectly_labeled_indexes]
                    if y_pred.shape[0] > 0:
                        MI_attack_blind_per_class_incorrectly_labeled[j] = balanced_accuracy_score(temp_y, y_pred)
                        MI_attack_blind_acc_per_class_incorrectly_labeled[j] = accuracy_score(temp_y, y_pred)
                        MI_attack_blind_far_per_class_incorrectly_labeled[j] = false_alarm_rate(temp_y, y_pred)
                        MI_attack_blind_prec_per_class_incorrectly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                        MI_attack_blind_rcal_per_class_incorrectly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                        MI_attack_blind_f1_per_class_incorrectly_labeled[j] = f1_score(temp_y, y_pred, average=None)

                if verbose:
                    print('\nMI Naive Attack (correctly Classified):', MI_attack_blind_per_class_correctly_labeled[j],
                          np.sum(MI_correctly_labeled_indexes), np.sum(MI_y_test[MI_correctly_labeled_indexes] == 0),
                          np.sum(MI_y_test[MI_correctly_labeled_indexes] == 1))
                    print('MI Naive Attack(FAR):', MI_attack_blind_far_per_class_correctly_labeled[j])
                    print('MI Naive Attack(Acc-unbalanced):', MI_attack_blind_acc_per_class_correctly_labeled[j])
                    print('MI Naive Attack(Prec):', MI_attack_blind_prec_per_class_correctly_labeled[j])
                    print('MI Naive Attack(Rec):', MI_attack_blind_rcal_per_class_correctly_labeled[j])
                    print('MI Naive Attack(F1):', MI_attack_blind_f1_per_class_correctly_labeled[j])
                    print('\nMI Naive Attack (Misclassified):', MI_attack_blind_per_class_incorrectly_labeled[j],
                          np.sum(MI_incorrectly_labeled_indexes), np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 0),
                          np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 1))
                    print('MI Naive Attack(FAR):', MI_attack_blind_far_per_class_incorrectly_labeled[j])
                    print('MI Naive Attack(Acc-unbalanced):', MI_attack_blind_acc_per_class_incorrectly_labeled[j])
                    print('MI Naive Attack(Prec):', MI_attack_blind_prec_per_class_incorrectly_labeled[j])
                    print('MI Naive Attack(Rec):', MI_attack_blind_rcal_per_class_incorrectly_labeled[j])
                    print('MI Naive Attack(F1):', MI_attack_blind_f1_per_class_incorrectly_labeled[j])

    if show_MI_attack:
        MI_attack, MI_attack_std = average_over_positive_values(MI_attack_per_class)
        MI_attack_correct_only, MI_attack_correct_only_std = average_over_positive_values(MI_attack_per_class_correctly_labeled)
        MI_attack_incorrect_only, MI_attack_incorrect_only_std = average_over_positive_values(MI_attack_per_class_incorrectly_labeled)

        MI_attack_acc, MI_attack_acc_std = average_over_positive_values(MI_attack_acc_per_class)
        MI_attack_acc_correct_only, MI_attack_acc_correct_only_std = average_over_positive_values(MI_attack_acc_per_class_correctly_labeled)
        MI_attack_acc_incorrect_only, MI_attack_acc_incorrect_only_std = average_over_positive_values(MI_attack_acc_per_class_incorrectly_labeled)

        MI_attack_far, MI_attack_far_std = average_over_positive_values(MI_attack_far_per_class)
        MI_attack_far_correct_only, MI_attack_far_correct_only_std = average_over_positive_values(MI_attack_far_per_class_correctly_labeled)
        MI_attack_far_incorrect_only, MI_attack_far_incorrect_only_std = average_over_positive_values(MI_attack_far_per_class_incorrectly_labeled)

        MI_attack_prec, MI_attack_prec_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class)
        MI_attack_prec_correct_only, MI_attack_prec_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_correctly_labeled)
        MI_attack_prec_incorrect_only, MI_attack_prec_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_incorrectly_labeled)

        MI_attack_rcal, MI_attack_rcal_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class)
        MI_attack_rcal_correct_only, MI_attack_rcal_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_correctly_labeled)
        MI_attack_rcal_incorrect_only, MI_attack_rcal_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_incorrectly_labeled)

        MI_attack_f1, MI_attack_f1_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class)
        MI_attack_f1_correct_only, MI_attack_f1_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_correctly_labeled)
        MI_attack_f1_incorrect_only, MI_attack_f1_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_incorrectly_labeled)

    if show_blind_attack:
        MI_attack_blind, MI_attack_blind_std = average_over_positive_values(MI_attack_blind_per_class)
        MI_attack_blind_correct_only, MI_attack_blind_correct_only_std = average_over_positive_values(MI_attack_blind_per_class_correctly_labeled)
        MI_attack_blind_incorrect_only, MI_attack_blind_incorrect_only_std = average_over_positive_values(MI_attack_blind_per_class_incorrectly_labeled)

        MI_attack_blind_acc, MI_attack_blind_acc_std = average_over_positive_values(MI_attack_blind_acc_per_class)
        MI_attack_blind_acc_correct_only, MI_attack_blind_acc_correct_only_std = average_over_positive_values(MI_attack_blind_acc_per_class_correctly_labeled)
        MI_attack_blind_acc_incorrect_only, MI_attack_blind_acc_incorrect_only_std = average_over_positive_values(MI_attack_blind_acc_per_class_incorrectly_labeled)

        MI_attack_blind_far, MI_attack_blind_far_std = average_over_positive_values(MI_attack_blind_far_per_class)
        MI_attack_blind_far_correct_only, MI_attack_blind_far_correct_only_std = average_over_positive_values(MI_attack_blind_far_per_class_correctly_labeled)
        MI_attack_blind_far_incorrect_only, MI_attack_blind_far_incorrect_only_std = average_over_positive_values(MI_attack_blind_far_per_class_incorrectly_labeled)

        MI_attack_blind_prec, MI_attack_blind_prec_std = average_over_positive_values_of_2d_array(MI_attack_blind_prec_per_class)
        MI_attack_blind_prec_correct_only, MI_attack_blind_prec_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_blind_prec_per_class_correctly_labeled)
        MI_attack_blind_prec_incorrect_only, MI_attack_blind_prec_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_blind_prec_per_class_incorrectly_labeled)

        MI_attack_blind_rcal, MI_attack_blind_rcal_std = average_over_positive_values_of_2d_array(MI_attack_blind_rcal_per_class)
        MI_attack_blind_rcal_correct_only, MI_attack_blind_rcal_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_blind_rcal_per_class_correctly_labeled)
        MI_attack_blind_rcal_incorrect_only, MI_attack_blind_rcal_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_blind_rcal_per_class_incorrectly_labeled)

        MI_attack_blind_f1, MI_attack_blind_f1_std = average_over_positive_values_of_2d_array(MI_attack_blind_f1_per_class)
        MI_attack_blind_f1_correct_only, MI_attack_blind_f1_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_blind_f1_per_class_correctly_labeled)
        MI_attack_blind_f1_incorrect_only, MI_attack_blind_f1_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_blind_f1_per_class_incorrectly_labeled)

    print("\nTarget model accuracy [train test]:")
    print(str(np.round(acc_train*100, 2)), str(np.round(acc_test*100, 2)))
    print("\nTarget model confidence [average standard_deviation]:")
    print(str(np.round(conf_train*100, 2)), str(np.round(conf_train_std*100, 2)), str(np.round(conf_test*100, 2)), str(np.round(conf_test_std*100, 2)))
    if report_separated_performance:
        print(str(np.round(conf_train_correct_only*100, 2)), str(np.round(conf_train_correct_only_std*100, 2)), str(np.round(conf_test_correct_only*100, 2)), str(np.round(conf_test_correct_only_std*100, 2)))
        print(str(np.round(conf_train_incorrect_only*100, 2)), str(np.round(conf_train_incorrect_only_std*100, 2)), str(np.round(conf_test_incorrect_only*100, 2)), str(np.round(conf_test_incorrect_only_std*100, 2)))

    if show_MI_attack:
        print("\n\n\nMI Attack bal. accuracy [average standard_deviation]:")
        print(str(np.round(MI_attack*100, 2)), str(np.round(MI_attack_std*100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_correct_only*100, 2)), str(np.round(MI_attack_correct_only_std*100, 2)), str(np.round(MI_attack_incorrect_only*100, 2)), str(np.round(MI_attack_incorrect_only_std*100, 2)))

        print("\n\n\nMI Attack FAR [average standard_deviation]:")
        print(str(np.round(MI_attack_far*100, 2)), str(np.round(MI_attack_far_std*100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_far_correct_only*100, 2)), str(np.round(MI_attack_far_correct_only_std*100, 2)), str(np.round(MI_attack_far_incorrect_only*100, 2)), str(np.round(MI_attack_far_incorrect_only_std*100, 2)))

        print("\n\n\nMI Attack unbal. accuracy [average standard_deviation]:")
        print(str(np.round(MI_attack_acc*100, 2)), str(np.round(MI_attack_acc_std*100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_acc_correct_only*100, 2)), str(np.round(MI_attack_acc_correct_only_std*100, 2)), str(np.round(MI_attack_acc_incorrect_only*100, 2)), str(np.round(MI_attack_acc_incorrect_only_std*100, 2)))

        print("\nMI Attack precision [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
        print(str(np.round(MI_attack_prec*100, 2)), str(np.round(MI_attack_prec_std*100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_prec_correct_only*100, 2)), str(np.round(MI_attack_prec_correct_only_std*100, 2)), str(np.round(MI_attack_prec_incorrect_only*100, 2)), str(np.round(MI_attack_prec_incorrect_only_std*100, 2)))

        print("\nMI Attack recall [[average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
        print(str(np.round(MI_attack_rcal*100, 2)), str(np.round(MI_attack_rcal_std*100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_rcal_correct_only*100, 2)), str(np.round(MI_attack_rcal_correct_only_std*100, 2)), str(np.round(MI_attack_rcal_incorrect_only*100, 2)), str(np.round(MI_attack_rcal_incorrect_only_std*100, 2)))

        print("\nMI Attack f1 [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
        print(str(np.round(MI_attack_f1*100, 2)), str(np.round(MI_attack_f1_std*100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_f1_correct_only*100, 2)), str(np.round(MI_attack_f1_correct_only_std*100, 2)), str(np.round(MI_attack_f1_incorrect_only*100, 2)), str(np.round(MI_attack_f1_incorrect_only_std*100, 2)))

    if show_blind_attack:
        print("\n\n\nMI Naive Attack accuracy [average standard_deviation]:")
        print(str(np.round(MI_attack_blind * 100, 2)), str(np.round(MI_attack_blind_std * 100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_blind_correct_only * 100, 2)), str(np.round(MI_attack_blind_correct_only_std * 100, 2)), str(np.round(MI_attack_blind_incorrect_only * 100, 2)), str(np.round(MI_attack_blind_incorrect_only_std * 100, 2)))

        print("\n\n\nMI Naive Attack FAR [average standard_deviation]:")
        print(str(np.round(MI_attack_blind_far * 100, 2)), str(np.round(MI_attack_blind_far_std * 100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_blind_far_correct_only * 100, 2)), str(np.round(MI_attack_blind_far_correct_only_std * 100, 2)), str(np.round(MI_attack_blind_far_incorrect_only * 100, 2)), str(np.round(MI_attack_blind_far_incorrect_only_std * 100, 2)))

        print("\n\n\nMI Naive Attack unbal. accuracy [average standard_deviation]:")
        print(str(np.round(MI_attack_blind_acc * 100, 2)), str(np.round(MI_attack_blind_acc_std * 100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_blind_acc_correct_only * 100, 2)), str(np.round(MI_attack_blind_acc_correct_only_std * 100, 2)), str(np.round(MI_attack_blind_acc_incorrect_only * 100, 2)), str(np.round(MI_attack_blind_acc_incorrect_only_std * 100, 2)))

        print("\nMI Naive Attack precision [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
        print(str(np.round(MI_attack_blind_prec * 100, 2)), str(np.round(MI_attack_blind_prec_std * 100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_blind_prec_correct_only * 100, 2)), str(np.round(MI_attack_blind_prec_correct_only_std * 100, 2)), str(np.round(MI_attack_blind_prec_incorrect_only * 100, 2)), str(np.round(MI_attack_blind_prec_incorrect_only_std * 100, 2)))

        print("\nMI Naive Attack recall [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
        print(str(np.round(MI_attack_blind_rcal * 100, 2)), str(np.round(MI_attack_blind_rcal_std * 100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_blind_rcal_correct_only * 100, 2)), str(np.round(MI_attack_blind_rcal_correct_only_std * 100, 2)), str(np.round(MI_attack_blind_rcal_incorrect_only * 100, 2)), str(np.round(MI_attack_blind_rcal_incorrect_only_std * 100, 2)))

        print("\nMI Naive Attack f1 [average(negative_class) average(positive_class)] [standard_deviation(negative) standard_deviation(positive_class)]:")
        print(str(np.round(MI_attack_blind_f1 * 100, 2)), str(np.round(MI_attack_blind_f1_std * 100, 2)))
        if report_separated_performance:
            print(str(np.round(MI_attack_blind_f1_correct_only * 100, 2)), str(np.round(MI_attack_blind_f1_correct_only_std * 100, 2)), str(np.round(MI_attack_blind_f1_incorrect_only * 100, 2)), str(np.round(MI_attack_blind_f1_incorrect_only_std * 100, 2)))

