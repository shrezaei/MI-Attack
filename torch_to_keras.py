from bearpaw.models.cifar.alexnet import AlexNet
from bearpaw.models.cifar.densenet import DenseNet
from bearpaw.models.cifar.resnet import ResNet
import pytorch2keras as p2k
import argparse
import torch
import keras
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
import tensorflow as tf


parser = argparse.ArgumentParser(description='To convert a Torch model to a Keras model.')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Indicate dataset. The bearpaw package only contains cifar10 and cifar100. For other models, you do not need a conversion. You can use the train_target_model to train a keras model from scratch.')
parser.add_argument('-m', '--model_name', type=str, default='alexnet', choices=['alexnet', 'resnet-110', 'densenet-bc-100-12', 'densenet-bc-L190-k40'], help='Indicate the model type used in bearpaw package.')
parser.add_argument('-t', '--torch_model_path', type=str, default='torch_models/cifar10/alexnet.pth.tar', help='Indicate the path to load the torch model.')
parser.add_argument('-k', '--keras_model_path', type=str, default='keras_models/cifar10/alexnet.h5', help='Indicate the path to save the keras model.')


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    model_name = args.model_name
    torch_model_path = args.torch_model_path
    save_path = args.keras_model_path

    if dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100

    if model_name == 'alexnet':
        torch_model = AlexNet(num_classes=num_classes)
    elif model_name == 'resnet-110':
        torch_model = ResNet(depth=164, num_classes=num_classes, block_name='bottleNeck')
    elif model_name == 'densenet-bc-100-12':
        torch_model = DenseNet(depth=100, num_classes=num_classes)
    elif model_name == 'densenet-bc-L190-k40':
        torch_model = DenseNet(depth=190, growthRate=40, num_classes=num_classes)
    else:
        print("Model name is unknown!")
        exit()

    checkpoint = torch.load(torch_model_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        print(k)
        # if 'module' not in k:
        #     k = 'module.' + k
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
    input_var = Variable(torch.FloatTensor(input_np))

    k_model = p2k.pytorch_to_keras(torch_model, input_var, [(3, 32, 32,)], verbose=True)
    k_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    k_model.summary()


    flat1 = k_model.layers[-1].output
    output = tf.keras.layers.Activation('softmax')(flat1)
    model = tf.keras.models.Model(inputs=k_model.inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.save(save_path)
