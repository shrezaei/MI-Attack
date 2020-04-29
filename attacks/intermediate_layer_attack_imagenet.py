from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from matplotlib import rcParams
from utils import average_over_positive_values, average_over_positive_values_of_2d_array, wigthed_average, load_Data_with_imagenet_id
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

rcParams.update({'font.size': 16})
# skip attack model training if there is no correctly labeled samples
cor_skip_threshold = 15
# skip attack model training if there is no incorrectly labeled samples
incor_skip_threshold = 10

def intermediate_layer_attack_imagenet(dataset, intermediate_layer, attack_classifier, sampling, what_portion_of_samples_attacker_knows, num_classes, num_targeted_classes, model_name, verbose, show_MI_attack, show_MI_attack_separate_result, show_MI_attack_separate_result_for_incorrect, imagenet_path):
    model = keras.models.load_model(model_name)
    if intermediate_layer == -1:
        AV_layer_output = model.layers[-2].output
    else:
        print('Error: You can onyl indicate -1 as an intermediate layer for InceptionV3 or Xception!')
        exit()

    intermediate_model = keras.models.Model(inputs=model.input, outputs=AV_layer_output)


    #Orinigal target model performance
    conf_train = np.zeros(num_targeted_classes) - 1
    conf_train_std = np.zeros(num_targeted_classes) - 1
    conf_test = np.zeros(num_targeted_classes) - 1
    conf_test_std = np.zeros(num_targeted_classes) - 1

    conf_train_correct_only = np.zeros(num_targeted_classes) - 1
    conf_train_correct_only_std = np.zeros(num_targeted_classes) - 1
    conf_train_incorrect_only = np.zeros(num_targeted_classes) - 1
    conf_train_incorrect_only_std = np.zeros(num_targeted_classes) - 1

    conf_test_correct_only = np.zeros(num_targeted_classes) - 1
    conf_test_correct_only_std = np.zeros(num_targeted_classes) - 1
    conf_test_incorrect_only = np.zeros(num_targeted_classes) - 1
    conf_test_incorrect_only_std = np.zeros(num_targeted_classes) - 1

    train_acc = np.zeros(num_targeted_classes) - 1
    test_acc = np.zeros(num_targeted_classes) - 1
    train_samples = np.zeros(num_targeted_classes) - 1
    test_samples = np.zeros(num_targeted_classes) - 1





    #To store per-class MI attack accuracy
    MI_attack_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

    MI_attack_auc_per_class = np.zeros(num_targeted_classes) - 1
    MI_attack_auc_per_class_correctly_labeled = np.zeros(num_targeted_classes) - 1
    MI_attack_auc_per_class_incorrectly_labeled = np.zeros(num_targeted_classes) - 1

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
    MI_attack_prec_per_class_correctly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_correctly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_correctly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1

    #The performance of attack on its training set. To see if it can learn anything
    MI_attack_per_class_correctly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_correctly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_correctly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_correctly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1

    MI_attack_per_class_incorrectly_labeled_separate = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_incorrectly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_incorrectly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_incorrectly_labeled_separate = np.zeros((num_targeted_classes, 2)) - 1

    #The performance of attack on its training set. To see if it can learn anything
    MI_attack_per_class_incorrectly_labeled_separate2 = np.zeros(num_targeted_classes) - 1
    MI_attack_prec_per_class_incorrectly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_rcal_per_class_incorrectly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1
    MI_attack_f1_per_class_incorrectly_labeled_separate2 = np.zeros((num_targeted_classes, 2)) - 1






    for j in range(num_targeted_classes):

        skip_attack_on_correctly_labeled = False
        skip_attack_on_incorrectly_labeled = False

        (x_train, y_train), (x_test, y_test), keras_class_id = load_Data_with_imagenet_id(j+1, imagenet_path=imagenet_path)

        # x_train = x_train[:200]
        # y_train = y_train[:200]

        x_train = keras.applications.inception_v3.preprocess_input(x_train)
        x_test = keras.applications.inception_v3.preprocess_input(x_test)
        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)
        train_samples[j] = x_train.shape[0]
        test_samples[j] = x_test.shape[0]

        confidence_train = model.predict(x_train)
        confidence_test = model.predict(x_test)
        print(confidence_train.shape, confidence_test.shape)
        labels_train_by_model = np.argmax(confidence_train, axis=1)
        labels_test_by_model = np.argmax(confidence_test, axis=1)
        print(labels_train_by_model.shape, labels_test_by_model.shape)

        intermediate_value_train = intermediate_model.predict(x_train)
        intermediate_value_test = intermediate_model.predict(x_test)
        attack_input_dimension = intermediate_value_train.shape[1]
        print(intermediate_value_train.shape, intermediate_value_test.shape, attack_input_dimension)

        train_acc[j] = accuracy_score(y_train, labels_train_by_model)
        test_acc[j] = accuracy_score(y_test, labels_test_by_model)
        print(train_acc[j], test_acc[j])

        labels_train = y_train
        labels_test = y_test
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        conf_train[j] = np.average(confidence_train[:, keras_class_id])
        conf_train_std[j] = np.std(confidence_train[:, keras_class_id])

        correctly_classified_indexes_train = labels_train_by_model == labels_train
        if np.sum(correctly_classified_indexes_train) > 0:
            conf_train_correct_only[j] = np.average(confidence_train[correctly_classified_indexes_train, keras_class_id])
            conf_train_correct_only_std[j] = np.std(confidence_train[correctly_classified_indexes_train, keras_class_id])

        incorrectly_classified_indexes_train = labels_train_by_model != labels_train
        if np.sum(incorrectly_classified_indexes_train) > 0:
            conf_train_incorrect_only[j] = np.average(confidence_train[incorrectly_classified_indexes_train, keras_class_id])
            conf_train_incorrect_only_std[j] = np.std(confidence_train[incorrectly_classified_indexes_train, keras_class_id])

        # Compute average confidence for test set
        conf_test[j] = np.average(confidence_test[:, keras_class_id])
        conf_test_std[j] = np.std(confidence_test[:, keras_class_id])

        correctly_classified_indexes_test = labels_test_by_model == labels_test
        if np.sum(correctly_classified_indexes_test) > 0:
            conf_test_correct_only[j] = np.average(confidence_test[correctly_classified_indexes_test, keras_class_id])
            conf_test_correct_only_std[j] = np.std(confidence_test[correctly_classified_indexes_test, keras_class_id])

        incorrectly_classified_indexes_test = labels_test_by_model != labels_test
        if np.sum(incorrectly_classified_indexes_test) > 0:
            conf_test_incorrect_only[j] = np.average(confidence_test[incorrectly_classified_indexes_test, keras_class_id])
            conf_test_incorrect_only_std[j] = np.std(confidence_test[incorrectly_classified_indexes_test, keras_class_id])


        #Prepare the data for training and testing attack models (for all data and also correctly labeled samples)
        # class_yes_x = confidence_train[tuple([labels_train == j])]
        # class_no_x = confidence_test[tuple([labels_test == j])]
        class_yes_x = intermediate_value_train
        class_no_x = intermediate_value_test

        if class_yes_x.shape[0] < 20 or class_no_x.shape[0] < 20:
            print("Class " + str(j) + " doesn't have enough sample for training an attack model!")
            continue

        class_yes_x_correctly_labeled = correctly_classified_indexes_train
        class_no_x_correctly_labeled = correctly_classified_indexes_test

        class_yes_x_incorrectly_labeled = incorrectly_classified_indexes_train
        class_no_x_incorrectly_labeled = incorrectly_classified_indexes_test


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


        #preparing data to train an attack model for incorrectly labeled samples
        if show_MI_attack_separate_result:
            cor_class_yes_x = intermediate_value_train[correctly_classified_indexes_train]
            cor_class_no_x = intermediate_value_test[correctly_classified_indexes_test]
            # cor_class_yes_x = cor_class_yes_x[np.argmax(cor_class_yes_x, axis=1) == j]
            # cor_class_no_x = cor_class_no_x[np.argmax(cor_class_no_x, axis=1) == j]

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
            incor_class_yes_x = intermediate_value_train[incorrectly_classified_indexes_train]
            incor_class_no_x = intermediate_value_test[incorrectly_classified_indexes_test]
            # incor_class_yes_x = incor_class_yes_x[np.argmax(incor_class_yes_x, axis=1) == j]
            # incor_class_no_x = incor_class_no_x[np.argmax(incor_class_no_x, axis=1) == j]

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
            MI_attack_per_class[j] = balanced_accuracy_score(MI_y_test, y_pred)
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
                MI_attack_prec_per_class_incorrectly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                MI_attack_rcal_per_class_incorrectly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                MI_attack_f1_per_class_incorrectly_labeled[j] = f1_score(temp_y, y_pred, average=None)

            if verbose:
                print('\nMI Attack (all data):')
                print('Accuracy:', MI_attack_per_class[j])
                print('Precision:', MI_attack_prec_per_class[j])
                print('Recall:', MI_attack_rcal_per_class[j])
                print('F1:', MI_attack_f1_per_class[j])
                print('\nMI Attack (correctly classified samples):')
                print('Accuracy:', MI_attack_per_class_correctly_labeled[j])
                print('Precision:', MI_attack_prec_per_class_correctly_labeled[j])
                print('Recall:', MI_attack_rcal_per_class_correctly_labeled[j])
                print('F1:', MI_attack_f1_per_class_correctly_labeled[j])
                print('\nMI Attack (misclassified samples):')
                print('Accuracy:', MI_attack_per_class_incorrectly_labeled[j])
                print('Precision:', MI_attack_prec_per_class_incorrectly_labeled[j])
                print('Recall:', MI_attack_rcal_per_class_incorrectly_labeled[j])
                print('F1:', MI_attack_f1_per_class_incorrectly_labeled[j])

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
            MI_attack_prec_per_class_correctly_labeled_separate2[j] = precision_score(cor_MI_y_train, y_pred2, average=None)
            MI_attack_rcal_per_class_correctly_labeled_separate2[j] = recall_score(cor_MI_y_train, y_pred2, average=None)
            MI_attack_f1_per_class_correctly_labeled_separate2[j] = f1_score(cor_MI_y_train, y_pred2, average=None)

            # print('\nMI Attack train set (specific to correctly labeled):', j, MI_attack_per_class_correctly_labeled_separate2[j], cor_MI_x_train.shape[0])
            # print('MI Attack:', MI_attack_prec_per_class_correctly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_rcal_per_class_correctly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_f1_per_class_correctly_labeled_separate2[j])


            MI_attack_per_class_correctly_labeled_separate[j] = balanced_accuracy_score(cor_MI_y_test, y_pred)
            MI_attack_prec_per_class_correctly_labeled_separate[j] = precision_score(cor_MI_y_test, y_pred, average=None)
            MI_attack_rcal_per_class_correctly_labeled_separate[j] = recall_score(cor_MI_y_test, y_pred, average=None)
            MI_attack_f1_per_class_correctly_labeled_separate[j] = f1_score(cor_MI_y_test, y_pred, average=None)

            if verbose:
                print('\nMI Attack model trained only on correctly classified samples:')
                print('Accuracy:', MI_attack_per_class_correctly_labeled_separate[j])
                print('Precision:', MI_attack_prec_per_class_correctly_labeled_separate[j])
                print('Recall:', MI_attack_rcal_per_class_correctly_labeled_separate[j])
                print('F1:', MI_attack_f1_per_class_correctly_labeled_separate[j])

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
            MI_attack_prec_per_class_incorrectly_labeled_separate2[j] = precision_score(incor_MI_y_train, y_pred2, average=None)
            MI_attack_rcal_per_class_incorrectly_labeled_separate2[j] = recall_score(incor_MI_y_train, y_pred2, average=None)
            MI_attack_f1_per_class_incorrectly_labeled_separate2[j] = f1_score(incor_MI_y_train, y_pred2, average=None)

            # print('\nMI Attack train set (specific to incorrectly labeled):', j, MI_attack_per_class_incorrectly_labeled_separate2[j], incor_MI_x_train.shape[0])
            # print('MI Attack:', MI_attack_prec_per_class_incorrectly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_rcal_per_class_incorrectly_labeled_separate2[j])
            # print('MI Attack:', MI_attack_f1_per_class_incorrectly_labeled_separate2[j])

            MI_attack_per_class_incorrectly_labeled_separate[j] = balanced_accuracy_score(incor_MI_y_test, y_pred)
            MI_attack_prec_per_class_incorrectly_labeled_separate[j] = precision_score(incor_MI_y_test, y_pred, average=None)
            MI_attack_rcal_per_class_incorrectly_labeled_separate[j] = recall_score(incor_MI_y_test, y_pred, average=None)
            MI_attack_f1_per_class_incorrectly_labeled_separate[j] = f1_score(incor_MI_y_test, y_pred, average=None)

            if verbose:
                print('\nMI Attack model trained only on correctly classified samples:')
                print('Accuracy:', MI_attack_per_class_incorrectly_labeled_separate[j])
                print('Precision:', MI_attack_prec_per_class_incorrectly_labeled_separate[j])
                print('Recall:', MI_attack_rcal_per_class_incorrectly_labeled_separate[j])
                print('F1:', MI_attack_f1_per_class_incorrectly_labeled_separate[j])

    if show_MI_attack:
        MI_attack, MI_attack_std = average_over_positive_values(MI_attack_per_class)
        MI_attack_correct_only, MI_attack_correct_only_std = average_over_positive_values(MI_attack_per_class_correctly_labeled)
        MI_attack_incorrect_only, MI_attack_incorrect_only_std = average_over_positive_values(MI_attack_per_class_incorrectly_labeled)

        MI_attack_prec, MI_attack_prec_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class)
        MI_attack_prec_correct_only, MI_attack_prec_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_correctly_labeled)
        MI_attack_prec_incorrect_only, MI_attack_prec_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_incorrectly_labeled)

        MI_attack_rcal, MI_attack_rcal_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class)
        MI_attack_rcal_correct_only, MI_attack_rcal_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_correctly_labeled)
        MI_attack_rcal_incorrect_only, MI_attack_rcal_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_incorrectly_labeled)

        MI_attack_f1, MI_attack_f1_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class)
        MI_attack_f1_correct_only, MI_attack_f1_correct_only_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_correctly_labeled)
        MI_attack_f1_incorrect_only, MI_attack_f1_incorrect_only_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_incorrectly_labeled)

    if show_MI_attack_separate_result:
        MI_attack_correct_only_separate_model, MI_attack_correct_only_separate_model_std = average_over_positive_values(MI_attack_per_class_correctly_labeled_separate)
        MI_attack_prec_correct_only_separate_model, MI_attack_prec_correct_only_separate_model_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_correctly_labeled_separate)
        MI_attack_rcal_correct_only_separate_model, MI_attack_rcal_correct_only_separate_model_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_correctly_labeled_separate)
        MI_attack_f1_correct_only_separate_model, MI_attack_f1_correct_only_separate_model_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_correctly_labeled_separate)

        MI_attack_correct_only_separate_model2, MI_attack_correct_only_separate_model_std2 = average_over_positive_values(MI_attack_per_class_correctly_labeled_separate2)
        MI_attack_prec_correct_only_separate_model2, MI_attack_prec_correct_only_separate_model_std2 = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_correctly_labeled_separate2)
        MI_attack_rcal_correct_only_separate_model2, MI_attack_rcal_correct_only_separate_model_std2 = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_correctly_labeled_separate2)
        MI_attack_f1_correct_only_separate_model2, MI_attack_f1_correct_only_separate_model_std2 = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_correctly_labeled_separate2)

    if show_MI_attack_separate_result_for_incorrect:
        MI_attack_incorrect_only_separate_model, MI_attack_incorrect_only_separate_model_std = average_over_positive_values(MI_attack_per_class_incorrectly_labeled_separate)
        MI_attack_prec_incorrect_only_separate_model, MI_attack_prec_incorrect_only_separate_model_std = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_incorrectly_labeled_separate)
        MI_attack_rcal_incorrect_only_separate_model, MI_attack_rcal_incorrect_only_separate_model_std = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_incorrectly_labeled_separate)
        MI_attack_f1_incorrect_only_separate_model, MI_attack_f1_incorrect_only_separate_model_std = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_incorrectly_labeled_separate)

        MI_attack_incorrect_only_separate_model2, MI_attack_incorrect_only_separate_model_std2 = average_over_positive_values(MI_attack_per_class_incorrectly_labeled_separate2)
        MI_attack_prec_incorrect_only_separate_model2, MI_attack_prec_incorrect_only_separate_model_std2 = average_over_positive_values_of_2d_array(MI_attack_prec_per_class_incorrectly_labeled_separate2)
        MI_attack_rcal_incorrect_only_separate_model2, MI_attack_rcal_incorrect_only_separate_model_std2 = average_over_positive_values_of_2d_array(MI_attack_rcal_per_class_incorrectly_labeled_separate2)
        MI_attack_f1_incorrect_only_separate_model2, MI_attack_f1_incorrect_only_separate_model_std2 = average_over_positive_values_of_2d_array(MI_attack_f1_per_class_incorrectly_labeled_separate2)


    print("\n\n---------------------------------------")
    print("Final results:")
    print("Values are in a pair of average and standard deviation.")

    if show_MI_attack:
        print("\n\nMI Attack accuracy:")
        print('All data: ', str(np.round(MI_attack*100, 2)), str(np.round(MI_attack_std*100, 2)))
        print('Correctly classified samples: ', str(np.round(MI_attack_correct_only*100, 2)), str(np.round(MI_attack_correct_only_std*100, 2)))
        print('Misclassified samples: ', str(np.round(MI_attack_incorrect_only * 100, 2)), str(np.round(MI_attack_incorrect_only_std * 100, 2)))

        print("\nMI Attack precision:")
        print('All data: ', str(np.round(MI_attack_prec*100, 2)), str(np.round(MI_attack_prec_std*100, 2)))
        print('Correctly classified samples: ', str(np.round(MI_attack_prec_correct_only*100, 2)), str(np.round(MI_attack_prec_correct_only_std*100, 2)))
        print('Misclassified samples: ', str(np.round(MI_attack_prec_incorrect_only*100, 2)), str(np.round(MI_attack_prec_incorrect_only_std*100, 2)))

        print("\nMI Attack recall:")
        print('All data: ', str(np.round(MI_attack_rcal*100, 2)), str(np.round(MI_attack_rcal_std*100, 2)))
        print('Correctly classified samples: ', str(np.round(MI_attack_rcal_correct_only*100, 2)), str(np.round(MI_attack_rcal_correct_only_std*100, 2)))
        print('Misclassified samples: ', str(np.round(MI_attack_rcal_incorrect_only*100, 2)), str(np.round(MI_attack_rcal_incorrect_only_std*100, 2)))

        print("\nMI Attack f1:")
        print('All data: ', str(np.round(MI_attack_f1*100, 2)), str(np.round(MI_attack_f1_std*100, 2)))
        print('Correctly classified samples: ', str(np.round(MI_attack_f1_correct_only*100, 2)), str(np.round(MI_attack_f1_correct_only_std*100, 2)))
        print('Misclassified samples: ', str(np.round(MI_attack_f1_incorrect_only*100, 2)), str(np.round(MI_attack_f1_incorrect_only_std*100, 2)))

    if show_MI_attack_separate_result:
        # print("\nMI Attack accuracy, specific to correctly labeled samples (on its train set):")
        # print(str(np.round(MI_attack_correct_only_separate_model2*100, 2)), str(np.round(MI_attack_correct_only_separate_model_std2*100, 2)))
        # print(str(np.round(MI_attack_prec_correct_only_separate_model2*100, 2)), str(np.round(MI_attack_prec_correct_only_separate_model_std2*100, 2)))
        # print(str(np.round(MI_attack_rcal_correct_only_separate_model2*100, 2)), str(np.round(MI_attack_rcal_correct_only_separate_model_std2*100, 2)))
        # print(str(np.round(MI_attack_f1_correct_only_separate_model2*100, 2)), str(np.round(MI_attack_f1_correct_only_separate_model_std2*100, 2)))

        print("\nMI attack specific to correctly labeled samples:")
        print('Accuracy: ', str(np.round(MI_attack_correct_only_separate_model*100, 2)), str(np.round(MI_attack_correct_only_separate_model_std*100, 2)))
        print('Precision: ', str(np.round(MI_attack_prec_correct_only_separate_model*100, 2)), str(np.round(MI_attack_prec_correct_only_separate_model_std*100, 2)))
        print('Recall: ', str(np.round(MI_attack_rcal_correct_only_separate_model*100, 2)), str(np.round(MI_attack_rcal_correct_only_separate_model_std*100, 2)))
        print('F1: ', str(np.round(MI_attack_f1_correct_only_separate_model*100, 2)), str(np.round(MI_attack_f1_correct_only_separate_model_std*100, 2)))


    if show_MI_attack_separate_result_for_incorrect:
        # print("\nMI Attack accuracy, specific to ***incorrectly labeled samples (on its train set):")
        # print(str(np.round(MI_attack_incorrect_only_separate_model2*100, 2)), str(np.round(MI_attack_incorrect_only_separate_model_std2*100, 2)))
        # print(str(np.round(MI_attack_prec_incorrect_only_separate_model2*100, 2)), str(np.round(MI_attack_prec_incorrect_only_separate_model_std2*100, 2)))
        # print(str(np.round(MI_attack_rcal_incorrect_only_separate_model2*100, 2)), str(np.round(MI_attack_rcal_incorrect_only_separate_model_std2*100, 2)))
        # print(str(np.round(MI_attack_f1_incorrect_only_separate_model2*100, 2)), str(np.round(MI_attack_f1_incorrect_only_separate_model_std2*100, 2)))

        print("\nMI attack specific to incorrectly labeled samples:")
        print('Accuracy: ', str(np.round(MI_attack_incorrect_only_separate_model*100, 2)), str(np.round(MI_attack_incorrect_only_separate_model_std*100, 2)))
        print('Precision: ', str(np.round(MI_attack_prec_incorrect_only_separate_model*100, 2)), str(np.round(MI_attack_prec_incorrect_only_separate_model_std*100, 2)))
        print('Recall: ', str(np.round(MI_attack_rcal_incorrect_only_separate_model*100, 2)), str(np.round(MI_attack_rcal_incorrect_only_separate_model_std*100, 2)))
        print('F1: ', str(np.round(MI_attack_f1_incorrect_only_separate_model*100, 2)), str(np.round(MI_attack_f1_incorrect_only_separate_model_std*100, 2)))

