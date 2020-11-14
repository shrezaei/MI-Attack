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
from utils import average_over_positive_values, average_over_positive_values_of_2d_array, false_alarm_rate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold


#skip attack model training if there is not enough correctly labeled samples
cor_skip_threshold = 10
#skip attack model training if there is not enough incorrectly labeled samples
incor_skip_threshold = 5

def normalize_data(data, means, stds):
    data /= 255
    data[:, :, :, 0] -= means[0]
    data[:, :, :, 0] /= stds[0]
    data[:, :, :, 1] -= means[1]
    data[:, :, :, 1] /= stds[1]
    data[:, :, :, 2] -= means[2]
    data[:, :, :, 2] /= stds[2]
    return data

def intermediate_layer_attack(dataset, intermediate_layer, attack_classifier, sampling, what_portion_of_samples_attacker_knows, num_classes, num_targeted_classes, model_name, verbose, show_MI_attack, show_MI_attack_separate_result, show_MI_attack_separate_result_for_incorrect):

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

    if dataset == "mnist":
        if intermediate_layer == -1:
            AV_layer_output = model.layers[-2].output
        elif intermediate_layer == -2:
            AV_layer_output = model.layers[-3].output
        elif intermediate_layer == -3:
            AV_layer_output = model.layers[-4].output
        else:
            print("Unknown intermediate layer!")
            exit()
    elif dataset == "cifar10" or dataset == "cifar100":
        if intermediate_layer == -1:
            AV_layer_output = model.layers[-3].output


    train_stat = model.evaluate(x_train, y_train, verbose=0)
    test_stat = model.evaluate(x_test, y_test, verbose=0)

    acc_train = train_stat[1]
    loss_train = train_stat[0]

    acc_test = test_stat[1]
    loss_test = test_stat[0]

    print(acc_train, acc_test)

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

    MI_attack_per_class_correctly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_acc_per_class_correctly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_far_per_class_correctly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_correctly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_correctly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_correctly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1

    # The performance of attack on its training set. To see if it can learn anything
    MI_attack_per_class_correctly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_acc_per_class_correctly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_far_per_class_correctly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_correctly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_correctly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_correctly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1

    MI_attack_per_class_incorrectly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_acc_per_class_incorrectly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_far_per_class_incorrectly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_incorrectly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_incorrectly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_incorrectly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1

    # The performance of attack on its training set. To see if it can learn anything
    MI_attack_per_class_incorrectly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_acc_per_class_incorrectly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_far_per_class_incorrectly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_incorrectly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_incorrectly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_incorrectly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1


    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=AV_layer_output)
    intermediate_value_train = intermediate_model.predict(x_train)
    intermediate_value_test = intermediate_model.predict(x_test)
    attack_input_dimension = intermediate_value_train.shape[1]

    for j in range(num_targeted_classes):

        skip_attack_on_correctly_labeled = False
        skip_attack_on_incorrectly_labeled = False

        #Prepare the data for training and testing attack models (for all data and also correctly labeled samples)
        class_yes_x = intermediate_value_train[tuple([labels_train == j])]
        class_no_x = intermediate_value_test[tuple([labels_test == j])]

        if (class_yes_x.shape[0] < 20 or class_no_x.shape[0] < 20) and show_MI_attack:
            print("Class " + str(j) + " doesn't have enough sample for training an attack model!!")
            continue

        class_yes_x_correctly_labeled = correctly_classified_indexes_train[tuple([labels_train == j])]
        class_no_x_correctly_labeled = correctly_classified_indexes_test[tuple([labels_test == j])]

        class_yes_x_incorrectly_labeled = incorrectly_classified_indexes_train[tuple([labels_train == j])]
        class_no_x_incorrectly_labeled = incorrectly_classified_indexes_test[tuple([labels_test == j])]


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

        print('MI attack on class ', j)
        MI_x_train = np.concatenate((class_yes_x_train, class_no_x_train), axis=0)
        MI_y_train = np.concatenate((class_yes_y_train, class_no_y_train), axis=0)
        MI_x_test = np.concatenate((class_yes_x_test, class_no_x_test), axis=0)
        MI_y_test = np.concatenate((class_yes_y_test, class_no_y_test), axis=0)
        MI_correctly_labeled_indexes = np.concatenate((class_yes_x_correctly_labeled, class_no_x_correctly_labeled), axis=0)
        MI_incorrectly_labeled_indexes = np.concatenate((class_yes_x_incorrectly_labeled, class_no_x_incorrectly_labeled), axis=0)


        #preparing data to train an attack model for correctly labeled samples
        if show_MI_attack_separate_result:
            correctly_classified_indexes_train_of_this_class = np.logical_and(correctly_classified_indexes_train, labels_train == j)
            correctly_classified_indexes_test_of_this_class = np.logical_and(correctly_classified_indexes_test, labels_test == j)
            cor_class_yes_x = intermediate_value_train[correctly_classified_indexes_train_of_this_class]
            cor_class_no_x = intermediate_value_test[correctly_classified_indexes_test_of_this_class]

            if cor_class_yes_x.shape[0] < cor_skip_threshold or cor_class_no_x.shape[0] < cor_skip_threshold:
                print("Class " + str(j) + " doesn't have enough sample of correctly labeled for training an attack model!", cor_class_yes_x.shape[0], cor_class_no_x.shape[0])
                skip_attack_on_correctly_labeled = True



            cor_class_yes_size = int(cor_class_yes_x.shape[0] * what_portion_of_samples_attacker_knows)
            cor_class_no_size = int(cor_class_no_x.shape[0] * what_portion_of_samples_attacker_knows)

            cor_class_yes_x_train = cor_class_yes_x[:cor_class_yes_size]
            cor_class_yes_y_train = np.ones(cor_class_yes_x_train.shape[0])
            cor_class_yes_x_test = cor_class_yes_x[cor_class_yes_size:]
            cor_class_yes_y_test = np.ones(cor_class_yes_x_test.shape[0])

            cor_class_no_x_train = cor_class_no_x[:cor_class_no_size]
            cor_class_no_y_train = np.zeros(cor_class_no_x_train.shape[0])
            cor_class_no_x_test = cor_class_no_x[cor_class_no_size:]
            cor_class_no_y_test = np.zeros(cor_class_no_x_test.shape[0])


            y_size = cor_class_yes_x_train.shape[0]
            n_size = cor_class_no_x_train.shape[0]
            if sampling == "undersampling":
                if y_size > n_size:
                    cor_class_yes_x_train = cor_class_yes_x_train[:n_size]
                    cor_class_yes_y_train = cor_class_yes_y_train[:n_size]
                else:
                    cor_class_no_x_train = cor_class_no_x_train[:y_size]
                    cor_class_no_y_train = cor_class_no_y_train[:y_size]
            elif sampling == "oversampling":
                if y_size > n_size:
                    cor_class_no_x_train = np.tile(cor_class_no_x_train, (int(y_size / n_size), 1))
                    cor_class_no_y_train = np.zeros(cor_class_no_x_train.shape[0])
                else:
                    cor_class_yes_x_train = np.tile(cor_class_yes_x_train, (int(n_size / y_size), 1))
                    cor_class_yes_y_train = np.ones(cor_class_yes_x_train.shape[0])

            cor_MI_x_train = np.concatenate((cor_class_yes_x_train, cor_class_no_x_train), axis=0)
            cor_MI_y_train = np.concatenate((cor_class_yes_y_train, cor_class_no_y_train), axis=0)
            cor_MI_x_test = np.concatenate((cor_class_yes_x_test, cor_class_no_x_test), axis=0)
            cor_MI_y_test = np.concatenate((cor_class_yes_y_test, cor_class_no_y_test), axis=0)

        #preparing data to train an attack model for incorrectly labeled samples
        if show_MI_attack_separate_result_for_incorrect:

            incorrectly_classified_indexes_train_of_this_class = np.logical_and(incorrectly_classified_indexes_train, labels_train == j)
            incorrectly_classified_indexes_test_of_this_class = np.logical_and(incorrectly_classified_indexes_test, labels_test == j)
            incor_class_yes_x = intermediate_value_train[incorrectly_classified_indexes_train_of_this_class]
            incor_class_no_x = intermediate_value_test[incorrectly_classified_indexes_test_of_this_class]

            if incor_class_yes_x.shape[0] < incor_skip_threshold or incor_class_no_x.shape[0] < incor_skip_threshold:
                print("Class " + str(j) + " for inccorectly labeled dataset doesn't have enough sample for training an attack model!", incor_class_yes_x.shape[0], incor_class_no_x.shape[0])
                skip_attack_on_incorrectly_labeled = True


            incor_class_yes_size = int(incor_class_yes_x.shape[0] * what_portion_of_samples_attacker_knows)
            incor_class_no_size = int(incor_class_no_x.shape[0] * what_portion_of_samples_attacker_knows)

            incor_class_yes_x_train = incor_class_yes_x[:incor_class_yes_size]
            incor_class_yes_y_train = np.ones(incor_class_yes_x_train.shape[0])
            incor_class_yes_x_test = incor_class_yes_x[incor_class_yes_size:]
            incor_class_yes_y_test = np.ones(incor_class_yes_x_test.shape[0])

            incor_class_no_x_train = incor_class_no_x[:incor_class_no_size]
            incor_class_no_y_train = np.zeros(incor_class_no_x_train.shape[0])
            incor_class_no_x_test = incor_class_no_x[incor_class_no_size:]
            incor_class_no_y_test = np.zeros(incor_class_no_x_test.shape[0])


            y_size = incor_class_yes_x_train.shape[0]
            n_size = incor_class_no_x_train.shape[0]
            if sampling == "undersampling":
                if y_size > n_size:
                    incor_class_yes_x_train = incor_class_yes_x_train[:n_size]
                    incor_class_yes_y_train = incor_class_yes_y_train[:n_size]
                else:
                    incor_class_no_x_train = incor_class_no_x_train[:y_size]
                    incor_class_no_y_train = incor_class_no_y_train[:y_size]
            elif sampling == "oversampling":
                if y_size > n_size:
                    incor_class_no_x_train = np.tile(incor_class_no_x_train, (int(y_size / n_size), 1))
                    incor_class_no_y_train = np.zeros(incor_class_no_x_train.shape[0])
                else:
                    incor_class_yes_x_train = np.tile(incor_class_yes_x_train, (int(n_size / y_size), 1))
                    incor_class_yes_y_train = np.ones(incor_class_yes_x_train.shape[0])

            incor_MI_x_train = np.concatenate((incor_class_yes_x_train, incor_class_no_x_train), axis=0)
            incor_MI_y_train = np.concatenate((incor_class_yes_y_train, incor_class_no_y_train), axis=0)
            incor_MI_x_test = np.concatenate((incor_class_yes_x_test, incor_class_no_x_test), axis=0)
            incor_MI_y_test = np.concatenate((incor_class_yes_y_test, incor_class_no_y_test), axis=0)

        if show_MI_attack:
            if attack_classifier == "NN":
                # Use NN classifier to launch Membership Inference attack (All data + correctly labeled)
                attack_model = Sequential()
                attack_model.add(Dense(128, input_dim=attack_input_dimension, activation='relu'))
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
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
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
            if y_pred.shape[0] > 0:
                MI_attack_per_class[j] = balanced_accuracy_score(MI_y_test, y_pred)
                MI_attack_acc_per_class[j] = accuracy_score(MI_y_test, y_pred)
                MI_attack_far_per_class[j] = false_alarm_rate(MI_y_test, y_pred)
                MI_attack_prec_per_class[j] = precision_score(MI_y_test, y_pred, average=None)
                MI_attack_rcal_per_class[j] = recall_score(MI_y_test, y_pred, average=None)
                MI_attack_f1_per_class[j] = f1_score(MI_y_test, y_pred, average=None)

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
                incorrect_count = np.sum(temp_y == 1)
            if verbose:
                print('\nMI Attack (all):', MI_attack_per_class[j], MI_x_test.shape[0], np.sum([MI_y_train == 0]),
                      np.sum([MI_y_train == 1]), np.sum([MI_y_test == 0]), np.sum([MI_y_test == 1]))
                print('MI Attack(FAR):', MI_attack_far_per_class[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class[j])
                print('MI Attack(F1):', MI_attack_f1_per_class[j])
                print('\nMI Attack (correctly classified):', MI_attack_per_class_correctly_labeled[j],
                      np.sum(MI_correctly_labeled_indexes), np.sum(MI_y_test[MI_correctly_labeled_indexes] == 0),
                      np.sum(MI_y_test[MI_correctly_labeled_indexes] == 1))
                print('MI Attack(FAR):', MI_attack_far_per_class_correctly_labeled[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_correctly_labeled[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_correctly_labeled[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_correctly_labeled[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_correctly_labeled[j])
                print('\nMI Attack (Misclassified):', MI_attack_per_class_incorrectly_labeled[j],
                      np.sum(MI_incorrectly_labeled_indexes), np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 0),
                      np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 1))
                print('MI Attack(FAR):', MI_attack_far_per_class_incorrectly_labeled[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_incorrectly_labeled[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_incorrectly_labeled[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_incorrectly_labeled[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_incorrectly_labeled[j])


        # Use NN classifier to launch Membership Inference attack only on incorrectly labeled
        if show_MI_attack_separate_result and skip_attack_on_correctly_labeled == False:
            if attack_classifier == "NN":
                attack_model = Sequential()
                attack_model.add(Dense(64, input_dim=attack_input_dimension, activation='relu'))
                # attack_model.add(Dense(64, activation='relu'))
                attack_model.add(Dense(1, activation='sigmoid'))
                attack_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                attack_model.fit(cor_MI_x_train, cor_MI_y_train, epochs=30, batch_size=32, verbose=False)

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
                    RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=40, cv=2, verbose=1, random_state=42, n_jobs=-1)
                else:
                    RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=40, cv=2,
                                                    verbose=0, random_state=42, n_jobs=-1)
                RFR_random.fit(cor_MI_x_train, cor_MI_y_train)
                if verbose:
                    print(RFR_random.best_params_)
                attack_model = RFR_random.best_estimator_

            elif attack_classifier == "XGBoost":
                temp_model = XGBClassifier()
                param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100] , min_child_weight=[1, 5, 10, 15], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], max_depth=[3, 6, 9, 12])
                # param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100, 500, 1000])
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
                # grid = GridSearchCV(estimator=temp_model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
                grid = RandomizedSearchCV(estimator=temp_model, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
                grid_result = grid.fit(cor_MI_x_train, cor_MI_y_train)
                attack_model = grid_result.best_estimator_
                if verbose:
                    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

            if attack_classifier == "NN":
                y_pred = attack_model.predict_classes(cor_MI_x_test)
                y_pred2 = attack_model.predict_classes(cor_MI_x_train)
            else:
                y_pred = attack_model.predict(cor_MI_x_test)
                y_pred2 = attack_model.predict(cor_MI_x_train)

            MI_attack_per_class_correctly_labeled_separate2[j] = balanced_accuracy_score(cor_MI_y_train, y_pred2)
            MI_attack_acc_per_class_correctly_labeled_separate2[j] = accuracy_score(cor_MI_y_train, y_pred2)
            MI_attack_far_per_class_correctly_labeled_separate2[j] = false_alarm_rate(cor_MI_y_train, y_pred2)
            MI_attack_prec_per_class_correctly_labeled_separate2[j] = precision_score(cor_MI_y_train, y_pred2, average=None)
            MI_attack_rcal_per_class_correctly_labeled_separate2[j] = recall_score(cor_MI_y_train, y_pred2, average=None)
            MI_attack_f1_per_class_correctly_labeled_separate2[j] = f1_score(cor_MI_y_train, y_pred2, average=None)

            # print('\nMI Attack train set (specific to correctly labeled):', j, MI_attack_per_class_correctly_labeled_separate2[j], cor_MI_x_train.shape[0])
            # print('MI Attack:', MI_attack_prec_per_class_correctly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_rcal_per_class_correctly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_f1_per_class_correctly_labeled_separate2[j])

            MI_attack_per_class_correctly_labeled_separate[j] = balanced_accuracy_score(cor_MI_y_test, y_pred)
            MI_attack_acc_per_class_correctly_labeled_separate[j] = accuracy_score(cor_MI_y_test, y_pred)
            MI_attack_far_per_class_correctly_labeled_separate[j] = false_alarm_rate(cor_MI_y_test, y_pred)
            MI_attack_prec_per_class_correctly_labeled_separate[j] = precision_score(cor_MI_y_test, y_pred, average=None)
            MI_attack_rcal_per_class_correctly_labeled_separate[j] = recall_score(cor_MI_y_test, y_pred, average=None)
            MI_attack_f1_per_class_correctly_labeled_separate[j] = f1_score(cor_MI_y_test, y_pred, average=None)

            if verbose:
                print('\nMI Attack (specific to correctly labeled):', j,
                      MI_attack_per_class_correctly_labeled_separate[j], cor_MI_x_test.shape[0])
                print('MI Attack(FAR):', MI_attack_far_per_class_correctly_labeled_separate[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_correctly_labeled_separate[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_correctly_labeled_separate[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_correctly_labeled_separate[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_correctly_labeled_separate[j])

        # Use NN classifier to launch Membership Inference attack only on incorrectly labeled
        if show_MI_attack_separate_result_for_incorrect and skip_attack_on_incorrectly_labeled == False:
            if attack_classifier == "NN":
                attack_model = Sequential()
                attack_model.add(Dense(64, input_dim=attack_input_dimension, activation='relu'))
                # attack_model.add(Dense(64, activation='relu'))
                attack_model.add(Dense(1, activation='sigmoid'))
                attack_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                attack_model.fit(incor_MI_x_train, incor_MI_y_train, epochs=30, batch_size=32, verbose=False)

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
                RFR_random.fit(incor_MI_x_train, incor_MI_y_train)
                if verbose:
                    print(RFR_random.best_params_)
                attack_model = RFR_random.best_estimator_

            elif attack_classifier == "XGBoost":
                temp_model = XGBClassifier()
                param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100] , min_child_weight=[1, 5, 10, 15], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], max_depth=[3, 6, 9, 12])
                # param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100, 500, 1000])
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
                # grid = GridSearchCV(estimator=temp_model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
                grid = RandomizedSearchCV(estimator=temp_model, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
                grid_result = grid.fit(incor_MI_x_train, incor_MI_y_train)
                attack_model = grid_result.best_estimator_
                if verbose:
                    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


            if attack_classifier == "NN":
                y_pred = attack_model.predict_classes(incor_MI_x_test)
                y_pred2 = attack_model.predict_classes(incor_MI_x_train)
            else:
                y_pred = attack_model.predict(incor_MI_x_test)
                y_pred2 = attack_model.predict(incor_MI_x_train)

            MI_attack_per_class_incorrectly_labeled_separate2[j] = balanced_accuracy_score(incor_MI_y_train, y_pred2)
            MI_attack_acc_per_class_incorrectly_labeled_separate2[j] = accuracy_score(incor_MI_y_train, y_pred2)
            MI_attack_far_per_class_incorrectly_labeled_separate2[j] = false_alarm_rate(incor_MI_y_train, y_pred2)
            MI_attack_prec_per_class_incorrectly_labeled_separate2[j] = precision_score(incor_MI_y_train, y_pred2, average=None)
            MI_attack_rcal_per_class_incorrectly_labeled_separate2[j] = recall_score(incor_MI_y_train, y_pred2, average=None)
            MI_attack_f1_per_class_incorrectly_labeled_separate2[j] = f1_score(incor_MI_y_train, y_pred2, average=None)

            # print('\nMI Attack train set (specific to incorrectly labeled):', j, MI_attack_per_class_incorrectly_labeled_separate2[j], incor_MI_x_train.shape[0])
            # print('MI Attack:', MI_attack_prec_per_class_incorrectly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_rcal_per_class_incorrectly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_f1_per_class_incorrectly_labeled_separate2[j])

            MI_attack_per_class_incorrectly_labeled_separate[j] = balanced_accuracy_score(incor_MI_y_test, y_pred)
            MI_attack_acc_per_class_incorrectly_labeled_separate[j] = accuracy_score(incor_MI_y_test, y_pred)
            MI_attack_far_per_class_incorrectly_labeled_separate[j] = false_alarm_rate(incor_MI_y_test, y_pred)
            MI_attack_prec_per_class_incorrectly_labeled_separate[j] = precision_score(incor_MI_y_test, y_pred, average=None)
            MI_attack_rcal_per_class_incorrectly_labeled_separate[j] = recall_score(incor_MI_y_test, y_pred, average=None)
            MI_attack_f1_per_class_incorrectly_labeled_separate[j] = f1_score(incor_MI_y_test, y_pred, average=None)

            if verbose:
                print('\nMI Attack (specific to incorrectly labeled):', j,
                      MI_attack_per_class_incorrectly_labeled_separate[j], incor_MI_x_test.shape[0])
            print('MI Attack(FAR):', MI_attack_far_per_class_incorrectly_labeled_separate[j])
            print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_incorrectly_labeled_separate[j])
            print('MI Attack(Prec):', MI_attack_prec_per_class_incorrectly_labeled_separate[j])
            print('MI Attack(Rec):', MI_attack_rcal_per_class_incorrectly_labeled_separate[j])
            print('MI Attack(F1):', MI_attack_f1_per_class_incorrectly_labeled_separate[j])

    if show_MI_attack:
        MI_attack, MI_attack_std = average_over_positive_values(MI_attack_per_class)
        MI_attack_correct_only, MI_attack_correct_only_std = average_over_positive_values(
            MI_attack_per_class_correctly_labeled)
        MI_attack_incorrect_only, MI_attack_incorrect_only_std = average_over_positive_values(
            MI_attack_per_class_incorrectly_labeled)

        MI_attack_acc, MI_attack_acc_std = average_over_positive_values(MI_attack_acc_per_class)
        MI_attack_acc_correct_only, MI_attack_acc_correct_only_std = average_over_positive_values(
            MI_attack_acc_per_class_correctly_labeled)
        MI_attack_acc_incorrect_only, MI_attack_acc_incorrect_only_std = average_over_positive_values(
            MI_attack_acc_per_class_incorrectly_labeled)

        MI_attack_far, MI_attack_far_std = average_over_positive_values(MI_attack_far_per_class)
        MI_attack_far_correct_only, MI_attack_far_correct_only_std = average_over_positive_values(
            MI_attack_far_per_class_correctly_labeled)
        MI_attack_far_incorrect_only, MI_attack_far_incorrect_only_std = average_over_positive_values(
            MI_attack_far_per_class_incorrectly_labeled)

        MI_attack_prec, MI_attack_prec_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class)
        MI_attack_prec_correct_only, MI_attack_prec_correct_only_std = average_over_positive_values_of_2d_array(
            MI_attack_prec_per_class_correctly_labeled)
        MI_attack_prec_incorrect_only, MI_attack_prec_incorrect_only_std = average_over_positive_values_of_2d_array(
            MI_attack_prec_per_class_incorrectly_labeled)

        MI_attack_rcal, MI_attack_rcal_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class)
        MI_attack_rcal_correct_only, MI_attack_rcal_correct_only_std = average_over_positive_values_of_2d_array(
            MI_attack_rcal_per_class_correctly_labeled)
        MI_attack_rcal_incorrect_only, MI_attack_rcal_incorrect_only_std = average_over_positive_values_of_2d_array(
            MI_attack_rcal_per_class_incorrectly_labeled)

        MI_attack_f1, MI_attack_f1_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class)
        MI_attack_f1_correct_only, MI_attack_f1_correct_only_std = average_over_positive_values_of_2d_array(
            MI_attack_f1_per_class_correctly_labeled)
        MI_attack_f1_incorrect_only, MI_attack_f1_incorrect_only_std = average_over_positive_values_of_2d_array(
            MI_attack_f1_per_class_incorrectly_labeled)

    if show_MI_attack_separate_result:
        MI_attack_correct_only_separate_model, MI_attack_correct_only_separate_model_std = average_over_positive_values(
            MI_attack_per_class_correctly_labeled_separate)
        MI_attack_acc_correct_only_separate_model, MI_attack_acc_correct_only_separate_model_std = average_over_positive_values(
            MI_attack_acc_per_class_correctly_labeled_separate)
        MI_attack_far_correct_only_separate_model, MI_attack_far_correct_only_separate_model_std = average_over_positive_values(
            MI_attack_far_per_class_correctly_labeled_separate)
        MI_attack_prec_correct_only_separate_model, MI_attack_prec_correct_only_separate_model_std = average_over_positive_values_of_2d_array(
            MI_attack_prec_per_class_correctly_labeled_separate)
        MI_attack_rcal_correct_only_separate_model, MI_attack_rcal_correct_only_separate_model_std = average_over_positive_values_of_2d_array(
            MI_attack_rcal_per_class_correctly_labeled_separate)
        MI_attack_f1_correct_only_separate_model, MI_attack_f1_correct_only_separate_model_std = average_over_positive_values_of_2d_array(
            MI_attack_f1_per_class_correctly_labeled_separate)

        MI_attack_correct_only_separate_model2, MI_attack_correct_only_separate_model_std2 = average_over_positive_values(
            MI_attack_per_class_correctly_labeled_separate2)
        MI_attack_acc_correct_only_separate_model2, MI_attack_acc_correct_only_separate_model_std2 = average_over_positive_values(
            MI_attack_acc_per_class_correctly_labeled_separate2)
        MI_attack_far_correct_only_separate_model2, MI_attack_far_correct_only_separate_model_std2 = average_over_positive_values(
            MI_attack_far_per_class_correctly_labeled_separate2)
        MI_attack_prec_correct_only_separate_model2, MI_attack_prec_correct_only_separate_model_std2 = average_over_positive_values_of_2d_array(
            MI_attack_prec_per_class_correctly_labeled_separate2)
        MI_attack_rcal_correct_only_separate_model2, MI_attack_rcal_correct_only_separate_model_std2 = average_over_positive_values_of_2d_array(
            MI_attack_rcal_per_class_correctly_labeled_separate2)
        MI_attack_f1_correct_only_separate_model2, MI_attack_f1_correct_only_separate_model_std2 = average_over_positive_values_of_2d_array(
            MI_attack_f1_per_class_correctly_labeled_separate2)

    if show_MI_attack_separate_result_for_incorrect:
        MI_attack_incorrect_only_separate_model, MI_attack_incorrect_only_separate_model_std = average_over_positive_values(
            MI_attack_per_class_incorrectly_labeled_separate)
        MI_attack_acc_incorrect_only_separate_model, MI_attack_acc_incorrect_only_separate_model_std = average_over_positive_values(
            MI_attack_acc_per_class_incorrectly_labeled_separate)
        MI_attack_far_incorrect_only_separate_model, MI_attack_far_incorrect_only_separate_model_std = average_over_positive_values(
            MI_attack_far_per_class_incorrectly_labeled_separate)
        MI_attack_prec_incorrect_only_separate_model, MI_attack_prec_incorrect_only_separate_model_std = average_over_positive_values_of_2d_array(
            MI_attack_prec_per_class_incorrectly_labeled_separate)
        MI_attack_rcal_incorrect_only_separate_model, MI_attack_rcal_incorrect_only_separate_model_std = average_over_positive_values_of_2d_array(
            MI_attack_rcal_per_class_incorrectly_labeled_separate)
        MI_attack_f1_incorrect_only_separate_model, MI_attack_f1_incorrect_only_separate_model_std = average_over_positive_values_of_2d_array(
            MI_attack_f1_per_class_incorrectly_labeled_separate)

        MI_attack_incorrect_only_separate_model2, MI_attack_incorrect_only_separate_model_std2 = average_over_positive_values(
            MI_attack_per_class_incorrectly_labeled_separate2)
        MI_attack_acc_incorrect_only_separate_model2, MI_attack_acc_incorrect_only_separate_model_std2 = average_over_positive_values(
            MI_attack_acc_per_class_incorrectly_labeled_separate2)
        MI_attack_far_incorrect_only_separate_model2, MI_attack_far_incorrect_only_separate_model_std2 = average_over_positive_values(
            MI_attack_far_per_class_incorrectly_labeled_separate2)
        MI_attack_prec_incorrect_only_separate_model2, MI_attack_prec_incorrect_only_separate_model_std2 = average_over_positive_values_of_2d_array(
            MI_attack_prec_per_class_incorrectly_labeled_separate2)
        MI_attack_rcal_incorrect_only_separate_model2, MI_attack_rcal_incorrect_only_separate_model_std2 = average_over_positive_values_of_2d_array(
            MI_attack_rcal_per_class_incorrectly_labeled_separate2)
        MI_attack_f1_incorrect_only_separate_model2, MI_attack_f1_incorrect_only_separate_model_std2 = average_over_positive_values_of_2d_array(
            MI_attack_f1_per_class_incorrectly_labeled_separate2)

    print("\nModel accuracy:")
    print(str(np.round(acc_train * 100, 2)), str(np.round(acc_test * 100, 2)))

    if show_MI_attack:
        print("\n\n\nMI Attack accuracy:")
        print(str(np.round(MI_attack * 100, 2)), str(np.round(MI_attack_std * 100, 2)))
        if show_MI_attack_separate_result:
            print(str(np.round(MI_attack_correct_only * 100, 2)), str(np.round(MI_attack_correct_only_std * 100, 2)),
                  str(np.round(MI_attack_incorrect_only * 100, 2)), str(np.round(MI_attack_incorrect_only_std * 100, 2)))

        print("\n\n\nMI Attack FAR:")
        print(str(np.round(MI_attack_far * 100, 2)), str(np.round(MI_attack_far_std * 100, 2)))
        if show_MI_attack_separate_result:
            print(str(np.round(MI_attack_far_correct_only * 100, 2)),
              str(np.round(MI_attack_far_correct_only_std * 100, 2)),
              str(np.round(MI_attack_far_incorrect_only * 100, 2)),
              str(np.round(MI_attack_far_incorrect_only_std * 100, 2)))

        print("\n\n\nMI Attack unbal. accuracy:")
        print(str(np.round(MI_attack_acc * 100, 2)), str(np.round(MI_attack_acc_std * 100, 2)))
        if show_MI_attack_separate_result:
            print(str(np.round(MI_attack_acc_correct_only * 100, 2)),
              str(np.round(MI_attack_acc_correct_only_std * 100, 2)),
              str(np.round(MI_attack_acc_incorrect_only * 100, 2)),
              str(np.round(MI_attack_acc_incorrect_only_std * 100, 2)))

        print("\nMI Attack precision:")
        print(str(np.round(MI_attack_prec * 100, 2)), str(np.round(MI_attack_prec_std * 100, 2)))
        if show_MI_attack_separate_result:
            print(str(np.round(MI_attack_prec_correct_only * 100, 2)),
              str(np.round(MI_attack_prec_correct_only_std * 100, 2)),
              str(np.round(MI_attack_prec_incorrect_only * 100, 2)),
              str(np.round(MI_attack_prec_incorrect_only_std * 100, 2)))

        print("\nMI Attack recall:")
        print(str(np.round(MI_attack_rcal * 100, 2)), str(np.round(MI_attack_rcal_std * 100, 2)))
        if show_MI_attack_separate_result:
            print(str(np.round(MI_attack_rcal_correct_only * 100, 2)),
              str(np.round(MI_attack_rcal_correct_only_std * 100, 2)),
              str(np.round(MI_attack_rcal_incorrect_only * 100, 2)),
              str(np.round(MI_attack_rcal_incorrect_only_std * 100, 2)))

        print("\nMI Attack f1:")
        print(str(np.round(MI_attack_f1 * 100, 2)), str(np.round(MI_attack_f1_std * 100, 2)))
        if show_MI_attack_separate_result:
            print(str(np.round(MI_attack_f1_correct_only * 100, 2)), str(np.round(MI_attack_f1_correct_only_std * 100, 2)),
              str(np.round(MI_attack_f1_incorrect_only * 100, 2)),
              str(np.round(MI_attack_f1_incorrect_only_std * 100, 2)))

    if show_MI_attack_separate_result:
        print("\nMI Attack accuracy, specific to correctly labeled samples (on its train set):")
        print(str(np.round(MI_attack_correct_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_correct_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_far_correct_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_far_correct_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_acc_correct_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_acc_correct_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_prec_correct_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_prec_correct_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_rcal_correct_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_rcal_correct_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_f1_correct_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_f1_correct_only_separate_model_std2 * 100, 2)))

        print("\nMI Attack accuracy, specific to correctly labeled samples:")
        print(str(np.round(MI_attack_correct_only_separate_model * 100, 2)),
              str(np.round(MI_attack_correct_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_far_correct_only_separate_model * 100, 2)),
              str(np.round(MI_attack_far_correct_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_acc_correct_only_separate_model * 100, 2)),
              str(np.round(MI_attack_acc_correct_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_prec_correct_only_separate_model * 100, 2)),
              str(np.round(MI_attack_prec_correct_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_rcal_correct_only_separate_model * 100, 2)),
              str(np.round(MI_attack_rcal_correct_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_f1_correct_only_separate_model * 100, 2)),
              str(np.round(MI_attack_f1_correct_only_separate_model_std * 100, 2)))

    if show_MI_attack_separate_result_for_incorrect:
        print("\nMI Attack accuracy, specific to ***incorrectly labeled samples (on its train set):")
        print(str(np.round(MI_attack_incorrect_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_incorrect_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_far_incorrect_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_far_incorrect_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_acc_incorrect_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_acc_incorrect_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_prec_incorrect_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_prec_incorrect_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_rcal_incorrect_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_rcal_incorrect_only_separate_model_std2 * 100, 2)))
        print(str(np.round(MI_attack_f1_incorrect_only_separate_model2 * 100, 2)),
              str(np.round(MI_attack_f1_incorrect_only_separate_model_std2 * 100, 2)))

        print("\nMI Attack accuracy, specific to ***incorrectly labeled samples:")
        print(str(np.round(MI_attack_incorrect_only_separate_model * 100, 2)),
              str(np.round(MI_attack_incorrect_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_far_incorrect_only_separate_model * 100, 2)),
              str(np.round(MI_attack_far_incorrect_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_acc_incorrect_only_separate_model * 100, 2)),
              str(np.round(MI_attack_acc_incorrect_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_prec_incorrect_only_separate_model * 100, 2)),
              str(np.round(MI_attack_prec_incorrect_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_rcal_incorrect_only_separate_model * 100, 2)),
              str(np.round(MI_attack_rcal_incorrect_only_separate_model_std * 100, 2)))
        print(str(np.round(MI_attack_f1_incorrect_only_separate_model * 100, 2)),
              str(np.round(MI_attack_f1_incorrect_only_separate_model_std * 100, 2)))


