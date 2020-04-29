import numpy as np
import os
import json
from keras.preprocessing import image


def average_over_positive_values(a):
    positives = a[a != -1]
    avgs = np.average(positives)
    stds = np.std(positives)
    return avgs, stds


def average_over_positive_values_of_2d_array(a):
    positives = [a[a[:, 0] != -1, 0], a[a[:, 1] != -1, 1]]
    avgs = np.average(positives, axis=1)
    stds = np.std(positives, axis=1)
    return avgs, stds


def average_of_gradient_metrics(a):
    avgs = np.zeros(a.shape[1])
    stds = np.zeros(a.shape[1])
    for i in range(a.shape[1]):
        positives = a[a[:, i] != -1, i]
        avgs[i] = np.average(positives)
        stds[i] = np.std(positives)
    return avgs, stds


def average_of_gradient_metrics_of_2d_array(a):
    avgs = np.zeros((a.shape[1], 2))
    stds = np.zeros((a.shape[1], 2))
    for i in range(a.shape[1]):
        positives = [a[a[:, i, 0] != -1, i, 0], a[a[:, i, 1] != -1, i, 1]]
        avgs[i] = np.average(positives, axis=1)
        stds[i] = np.std(positives, axis=1)
    return avgs, stds


def wigthed_average(value, count):
    return np.sum(value[value != -1] * count[value != -1]) / np.sum(count[value != -1])


def wigthed_average_for_gradient_metrics(value, count):
    avgs = np.zeros(7)
    for i in range(value.shape[1]):
        avgs[i] = np.sum(value[value[:, i] != -1, i] * count[value[:, i] != -1]) / np.sum(count[value[:, i] != -1])
    return avgs


def average_over_gradient_metrics(a):
    avgs = np.average(a, axis=0)
    stds = np.std(a, axis=0)
    return avgs, stds


def wigthed_average_over_gradient_metrics(value, count):
    avgs = np.zeros(value.shape[1])
    for i in range(value.shape[1]):
        metric = value[:, i]
        avgs[i] = np.sum(metric[metric != -1] * count[metric != -1]) / np.sum(count[metric != -1])
    return avgs


def imagenet_to_keras_mapping(id):
    # The class with label 0 does not exist in original ImageNet dataset
    imagenetmapping = ["non-existence index"]
    kerasmapping = []

    f = open(os.path.join(os.path.dirname(__file__), 'map_clsloc.txt'), "r")
    for i in range(1000):
        imagenetmapping.append(f.readline().split(' ')[0])

    with open(os.path.join(os.path.dirname(__file__), 'imagenet_class_index.json')) as json_file:
        data = json.load(json_file)
        for i in range(1000):
            kerasmapping.append(data[str(i)][0])

    class_code = imagenetmapping[id]
    keraslabel = kerasmapping.index(class_code)
    return keraslabel


def keras_to_imagenet_mapping(id):
    # The class with label 0 does not exist in original ImageNet dataset
    imagenetmapping = ["non-existence index"]
    kerasmapping = []

    f = open(os.path.join(os.path.dirname(__file__), 'map_clsloc.txt'), "r")
    for i in range(1000):
        imagenetmapping.append(f.readline().split(' ')[0])

    with open(os.path.join(os.path.dirname(__file__), 'imagenet_class_index.json')) as json_file:
        data = json.load(json_file)
        for i in range(1000):
            kerasmapping.append(data[str(i)][0])

    class_code = kerasmapping[id]
    keraslabel = imagenetmapping.index(class_code)
    return keraslabel


def load_Data_with_keras_id(keras_class_id, target_size=(299, 299), imagenet_path='imagenet/'):
    imagenet_class = keras_to_imagenet_mapping(keras_class_id)
    train_path = imagenet_path + 'train/' + str(imagenet_class) + '/'
    test_path = imagenet_path + 'val/' + str(imagenet_class) + '/'
    img_list = []
    for folder, subs, files in os.walk(train_path):
        for file in files:
            filename = folder + "/" + file
            img = image.load_img(filename, target_size=target_size)
            img = image.img_to_array(img)
            img_list.append(img)
    if len(img_list) == 0:
        print('Error: The train folder in the imagenet path is empty (' + train_path + ')')
        exit()
    x_train = np.array(img_list)

    img_list = []
    for folder, subs, files in os.walk(test_path):
        for file in files:
            filename = folder + "/" + file
            img = image.load_img(filename, target_size=target_size)
            img = image.img_to_array(img)
            img_list.append(img)
    if len(img_list) == 0:
        print('Error: The test folder in the imagenet path is empty (' + test_path + ')')
        exit()
    x_test = np.array(img_list)

    y_train = np.ones(x_train.shape[0]) * keras_class_id
    y_test = np.ones(x_test.shape[0]) * keras_class_id
    return (x_train, y_train), (x_test, y_test), imagenet_class


def load_Data_with_imagenet_id(imagenet_class_id, target_size=(299, 299), imagenet_path='imagenet/'):
    keras_class_id = imagenet_to_keras_mapping(imagenet_class_id)
    train_path = imagenet_path + 'train/' + str(imagenet_class_id) + '/'
    test_path = imagenet_path + 'val/' + str(imagenet_class_id) + '/'
    img_list = []
    for folder, subs, files in os.walk(train_path):
        for file in files:
            filename = folder + "/" + file
            img = image.load_img(filename, target_size=target_size)
            img = image.img_to_array(img)
            img_list.append(img)
    if len(img_list) == 0:
        print('Error: The train folder in the imagenet path is empty (' + train_path + ')')
        exit()
    x_train = np.array(img_list)

    img_list = []
    for folder, subs, files in os.walk(test_path):
        for file in files:
            filename = folder + "/" + file
            img = image.load_img(filename, target_size=target_size)
            img = image.img_to_array(img)
            img_list.append(img)
    if len(img_list) == 0:
        print('Error: The test folder in the imagenet path is empty (' + test_path + ')')
        exit()
    x_test = np.array(img_list)

    y_train = np.ones(x_train.shape[0]) * keras_class_id
    y_test = np.ones(x_test.shape[0]) * keras_class_id
    return (x_train, y_train), (x_test, y_test), keras_class_id