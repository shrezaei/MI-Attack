from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
from matplotlib import rcParams
from utils import average_over_positive_values, average_over_positive_values_of_2d_array, wigthed_average, load_Data_with_imagenet_id, false_alarm_rate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

rcParams.update({'font.size': 16})
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

show_MI_attack = True
show_blind_attack = True

def conf_based_attack_imagenet(dataset, attack_classifier, sampling, what_portion_of_samples_attacker_knows, save_confidence_histogram, report_separated_performance, num_classes, num_targeted_classes, model_name, verbose, imagenet_path):
    model = keras.models.load_model(model_name)

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

    for j in range(num_targeted_classes):
    # for j in range(98,100):

        (x_train, y_train), (x_test, y_test), keras_class_id = load_Data_with_imagenet_id(j+1, imagenet_path=imagenet_path)

        x_train = keras.applications.inception_v3.preprocess_input(x_train)
        x_test = keras.applications.inception_v3.preprocess_input(x_test)
        train_samples[j] = x_train.shape[0]
        test_samples[j] = x_test.shape[0]

        confidence_train = model.predict(x_train)
        confidence_test = model.predict(x_test)
        labels_train_by_model = np.argmax(confidence_train, axis=1)
        labels_test_by_model = np.argmax(confidence_test, axis=1)

        train_acc[j] = accuracy_score(y_train, labels_train_by_model)
        test_acc[j] = accuracy_score(y_test, labels_test_by_model)

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

        class_yes_x = confidence_train
        class_no_x = confidence_test

        if class_yes_x.shape[0] < 20 or class_no_x.shape[0] < 20:
            print("Class " + str(j) + " doesn't have enough sample for training an attack model!")
            continue

        class_yes_x_correctly_labeled = correctly_classified_indexes_train
        class_no_x_correctly_labeled = correctly_classified_indexes_test

        class_yes_x_incorrectly_labeled = incorrectly_classified_indexes_train
        class_no_x_incorrectly_labeled = incorrectly_classified_indexes_test

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
            plt.savefig('figures/conf histogram/' + dataset + '/correct-' + str(j) + '.eps')
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
            plt.savefig('figures/conf histogram/' + dataset + '/misclassified-' + str(j) + '.eps')
            plt.close()

            temp = class_yes_x[class_yes_x_correctly_labeled]
            temp2 = class_no_x[class_no_x_correctly_labeled]
            bins = np.arange(101) / 100
            plt.style.use('seaborn-deep')
            n, bins, patches = plt.hist([temp[:, keras_class_id], temp2[:, keras_class_id]], bins, normed=1, alpha=1, label=['Train samples', 'Test samples'])
            plt.xlabel('Model Confidence')
            plt.ylabel('Probability (%)')
            plt.legend(loc='upper left')
            plt.savefig('figures/conf histogram/' + dataset + '/' + str(j) + '.eps')
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
        #
        #     cor_MI_x_train = np.concatenate((cor_class_yes_x_train, cor_class_no_x_train), axis=0)
        #     cor_MI_y_train = np.concatenate((cor_class_yes_y_train, cor_class_no_y_train), axis=0)
        #     cor_MI_x_test = np.concatenate((cor_class_yes_x_test, cor_class_no_x_test), axis=0)
        #     cor_MI_y_test = np.concatenate((cor_class_yes_y_test, cor_class_no_y_test), axis=0)

        if show_MI_attack:
            if attack_classifier == "NN":
                # Use NN classifier to launch Membership Inference attack (All data + correctly labeled)
                attack_model = Sequential()
                attack_model.add(Dense(128, input_dim=num_classes, activation='relu'))
                attack_model.add(Dense(64, activation='relu'))
                attack_model.add(Dense(1, activation='sigmoid'))
                attack_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-34), metrics=['acc'])
                attack_model.fit(MI_x_train, MI_y_train, validation_data=(MI_x_test, MI_y_test), epochs=40, batch_size=32, verbose=False, shuffle=True)
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

        if verbose:
            print('\nMI Attack (all):', MI_attack_per_class[j], MI_x_test.shape[0], np.sum([MI_y_train == 0]), np.sum([MI_y_train == 1]), np.sum([MI_y_test == 0]), np.sum([MI_y_test == 1]))
            print('MI Attack(FAR):', MI_attack_far_per_class[j])
            print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class[j])
            print('MI Attack(Prec):', MI_attack_prec_per_class[j])
            print('MI Attack(Rec):', MI_attack_rcal_per_class[j])
            print('MI Attack(F1):', MI_attack_f1_per_class[j])

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
                print('\nMI Attack (correct):', MI_attack_per_class_correctly_labeled[j], np.sum(MI_correctly_labeled_indexes), np.sum(MI_y_test[MI_correctly_labeled_indexes] == 0), np.sum(MI_y_test[MI_correctly_labeled_indexes] == 1))
                print('MI Attack(FAR):', MI_attack_far_per_class_correctly_labeled[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_correctly_labeled[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_correctly_labeled[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_correctly_labeled[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_correctly_labeled[j])
                print('\nMI Attack (incorrect):', MI_attack_per_class_incorrectly_labeled[j], np.sum(MI_incorrectly_labeled_indexes), np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 0), np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 1))
                print('MI Attack(FAR):', MI_attack_far_per_class_incorrectly_labeled[j])
                print('MI Attack(Acc-unbalanced):', MI_attack_acc_per_class_incorrectly_labeled[j])
                print('MI Attack(Prec):', MI_attack_prec_per_class_incorrectly_labeled[j])
                print('MI Attack(Rec):', MI_attack_rcal_per_class_incorrectly_labeled[j])
                print('MI Attack(F1):', MI_attack_f1_per_class_incorrectly_labeled[j])



        if show_blind_attack:
            # MI_x_train_blind = MI_x_train[:, j]     #To be fare, I just use the test test, to compare with other attack, so I comment it
            MI_x_test_blind = np.argmax(MI_x_test, axis=1)
            MI_predicted_y_test_blind = [1 if l==keras_class_id else 0 for l in MI_x_test_blind]
            MI_predicted_y_test_blind = np.array(MI_predicted_y_test_blind)

            # MI Naive attack accuracy on all data
            y_pred = MI_predicted_y_test_blind
            MI_attack_blind_per_class[j] = balanced_accuracy_score(MI_y_test, y_pred)
            MI_attack_blind_acc_per_class[j] = accuracy_score(MI_y_test, y_pred)
            MI_attack_blind_far_per_class[j] = false_alarm_rate(MI_y_test, y_pred)
            MI_attack_blind_prec_per_class[j] = precision_score(MI_y_test, y_pred, average=None)
            MI_attack_blind_rcal_per_class[j] = recall_score(MI_y_test, y_pred, average=None)
            MI_attack_blind_f1_per_class[j] = f1_score(MI_y_test, y_pred, average=None)
            if verbose:
                print('\nMI Naive Attack (all):', MI_attack_blind_per_class[j], MI_x_test.shape[0],
                      np.sum([MI_y_train == 0]), np.sum([MI_y_train == 1]), np.sum([MI_y_test == 0]),
                      np.sum([MI_y_test == 1]))
                print('MI Naive Attack(FAR):', MI_attack_blind_far_per_class[j])
                print('MI Naive Attack(Acc-unbalanced):', MI_attack_blind_acc_per_class[j])
                print('MI Naive Attack(Prec):', MI_attack_blind_prec_per_class[j])
                print('MI Naive Attack(Rec):', MI_attack_blind_rcal_per_class[j])
                print('MI Naive Attack(F1):', MI_attack_blind_f1_per_class[j])

            if report_separated_performance:
                # MI naive accuracy on correctly labeled
                temp_y = MI_y_test[MI_correctly_labeled_indexes]
                y_pred = MI_predicted_y_test_blind[MI_correctly_labeled_indexes]
                MI_attack_blind_per_class_correctly_labeled[j] = balanced_accuracy_score(temp_y, y_pred)
                MI_attack_blind_acc_per_class_correctly_labeled[j] = accuracy_score(temp_y, y_pred)
                MI_attack_blind_far_per_class_correctly_labeled[j] = false_alarm_rate(temp_y, y_pred)
                MI_attack_blind_prec_per_class_correctly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                MI_attack_blind_rcal_per_class_correctly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                MI_attack_blind_f1_per_class_correctly_labeled[j] = f1_score(temp_y, y_pred, average=None)

                # MI naive attack accuracy on incorrectly labeled
                temp_y = MI_y_test[MI_incorrectly_labeled_indexes]
                y_pred = MI_predicted_y_test_blind[MI_incorrectly_labeled_indexes]
                MI_attack_blind_per_class_incorrectly_labeled[j] = balanced_accuracy_score(temp_y, y_pred)
                MI_attack_blind_acc_per_class_incorrectly_labeled[j] = accuracy_score(temp_y, y_pred)
                MI_attack_blind_far_per_class_incorrectly_labeled[j] = false_alarm_rate(temp_y, y_pred)
                MI_attack_blind_prec_per_class_incorrectly_labeled[j] = precision_score(temp_y, y_pred, average=None)
                MI_attack_blind_rcal_per_class_incorrectly_labeled[j] = recall_score(temp_y, y_pred, average=None)
                MI_attack_blind_f1_per_class_incorrectly_labeled[j] = f1_score(temp_y, y_pred, average=None)

            if verbose:
                print('\nMI Naive Attack (correct):', MI_attack_blind_per_class_correctly_labeled[j],
                      np.sum(MI_correctly_labeled_indexes), np.sum(MI_y_test[MI_correctly_labeled_indexes] == 0),
                      np.sum(MI_y_test[MI_correctly_labeled_indexes] == 1))
                print('MI Naive Attack(FAR):', MI_attack_blind_far_per_class_correctly_labeled[j])
                print('MI Naive Attack(Acc-unbalanced):', MI_attack_blind_acc_per_class_correctly_labeled[j])
                print('MI Naive Attack(Prec):', MI_attack_blind_prec_per_class_correctly_labeled[j])
                print('MI Naive Attack(Rec):', MI_attack_blind_rcal_per_class_correctly_labeled[j])
                print('MI Naive Attack(F1):', MI_attack_blind_f1_per_class_correctly_labeled[j])
                print('\nMI Naive Attack (incorrect):', MI_attack_blind_per_class_incorrectly_labeled[j],
                      np.sum(MI_incorrectly_labeled_indexes), np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 0),
                      np.sum(MI_y_test[MI_incorrectly_labeled_indexes] == 1))
                print('MI Naive Attack(FAR):', MI_attack_blind_far_per_class_incorrectly_labeled[j])
                print('MI Naive Attack(Acc-unbalanced):', MI_attack_blind_acc_per_class_incorrectly_labeled[j])
                print('MI Naive Attack(Prec):', MI_attack_blind_prec_per_class_incorrectly_labeled[j])
                print('MI Naive Attack(Rec):', MI_attack_blind_rcal_per_class_incorrectly_labeled[j])
                print('MI Naive Attack(F1):', MI_attack_blind_f1_per_class_incorrectly_labeled[j])

        # # Use NN classifier to launch Membership Inference attack only on incorrectly labeled
        # if show_MI_attack_separate_result:
        #     if attack_classifier == "NN":
        #         attack_model = Sequential()
        #         attack_model.add(Dense(128, input_dim=num_classes, activation='relu'))
        #         attack_model.add(Dense(64, activation='relu'))
        #         attack_model.add(Dense(1, activation='sigmoid'))
        #         attack_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #         attack_model.fit(cor_MI_x_train, cor_MI_y_train, epochs=40, batch_size=32, verbose=False)
        #
        #     elif attack_classifier == "RF":
        #         n_est = [500, 800, 1500, 2500, 5000]
        #         max_f = ['auto', 'sqrt']
        #         max_depth = [20, 30, 40, 50]
        #         max_depth.append(None)
        #         min_samples_s = [2, 5, 10, 15, 20]
        #         min_samples_l = [1, 2, 5, 10, 15]
        #         grid_param = {'n_estimators': n_est,
        #                       'max_features': max_f,
        #                       'max_depth': max_depth,
        #                       'min_samples_split': min_samples_s,
        #                       'min_samples_leaf': min_samples_l}
        #         RFR = RandomForestClassifier(random_state=1)
        #         if verbose:
        #             RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=100, cv=2, verbose=1, random_state=42, n_jobs=-1)
        #         else:
        #             RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=100, cv=2, verbose=0, random_state=42, n_jobs=-1)
        #         RFR_random.fit(cor_MI_x_train, cor_MI_y_train)
        #         if verbose:
        #             print(RFR_random.best_params_)
        #         attack_model = RFR_random.best_estimator_
        #
        #     elif attack_classifier == "XGBoost":
        #         temp_model = XGBClassifier()
        #         param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100] , min_child_weight=[1, 5, 10, 15], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], max_depth=[3, 6, 9, 12])
        #         # param_grid = dict(scale_pos_weight=[1, 5, 10, 50, 100, 500, 1000])
        #         cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=2, random_state=1)
        #         # grid = GridSearchCV(estimator=temp_model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
        #         grid = RandomizedSearchCV(estimator=temp_model, param_distributions=param_grid, n_iter=50, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
        #         grid_result = grid.fit(cor_MI_x_train, cor_MI_y_train)
        #         attack_model = grid_result.best_estimator_
        #         if verbose:
        #             print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        #
        #     if attack_classifier == "NN":
        #         y_pred = attack_model.predict_classes(cor_MI_x_test)
        #     else:
        #         y_pred = attack_model.predict(cor_MI_x_test)
        #
        #     MI_attack_per_class_correctly_labeled_separate[j] = balanced_accuracy_score(cor_MI_y_test, y_pred)
        #     MI_attack_prec_per_class_correctly_labeled_separate[j] = precision_score(cor_MI_y_test, y_pred, average=None)
        #     MI_attack_rcal_per_class_correctly_labeled_separate[j] = recall_score(cor_MI_y_test, y_pred, average=None)
        #     MI_attack_f1_per_class_correctly_labeled_separate[j] = f1_score(cor_MI_y_test, y_pred, average=None)
        #     if verbose:
        #         print('\nMI Attack model trained only on correctly classified samples:')
        #         print('Accuracy:', MI_attack_per_class_correctly_labeled_separate[j])
        #         print('Precision:', MI_attack_prec_per_class_correctly_labeled_separate[j])
        #         print('Recall:', MI_attack_rcal_per_class_correctly_labeled_separate[j])
        #         print('F1:', MI_attack_f1_per_class_correctly_labeled_separate[j])


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


