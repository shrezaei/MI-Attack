import argparse
import os
from attacks.gradient_analysis import gradient_norms
from attacks.gradient_analysis_imagenet import gradient_norms_imagenet

parser = argparse.ArgumentParser(description='This script obtain and save the gradient norms of samples.')
parser.add_argument('-d', '--dataset', type=str, default='cifar_10', choices=['mnist', 'cifar_10', 'cifar_100', 'cifar_100_resnet', 'cifar_100_densenet', 'imagenet_inceptionv3', 'imagenet_xception'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-m', '--model_path', type=str, default='none', help='Indicate the path to the target model. If you used the train_target_model.py to train the model, leave this field to the default value.')
parser.add_argument('-n', '--number_of_target_classes', type=int, default=0, help='Limit the MI attack to limited a number of classes for efficiency!')
parser.add_argument('-s', '--num_of_samples_per_class', type=int, default=400, help='Due to the high computational complexity, you can specify how many samples per class to be analyzed.')
parser.add_argument('-i', '--imagenet_path', type=str, default='../imagenet/', help='path to the imagenet dataset.')
args = parser.parse_args()




if __name__ == '__main__':
    dataset = args.dataset
    num_of_samples_per_class = args.num_of_samples_per_class
    if num_of_samples_per_class <= 0:
        print('Error: num_of_samples_per_class should be a positive integer!')
        exit()

    gradient_save_dir = os.path.join(os.getcwd(), 'saved_gradients')
    if not os.path.isdir(gradient_save_dir):
        os.makedirs(gradient_save_dir)

    model_save_dir = os.path.join(os.getcwd(), 'saved_models')
    if dataset == "mnist" or dataset == "cifar_10":
        model_name = model_save_dir + '/' + dataset + '_weights_' + 'final.h5'
        num_classes = 10
        num_targeted_classes = 10
    elif dataset == "cifar_100" or dataset == "cifar_100_resnet" or dataset == "cifar_100_densenet":
        model_name = model_save_dir + '/' + dataset + '_weights_' + 'final.h5'
        num_classes = 100
        num_targeted_classes = 100
    elif dataset == "imagenet_inceptionv3":
        model_name = model_save_dir + "/imagenet_inceptionV3_v2.hdf5"
        num_classes = 1000
        num_targeted_classes = 100
    elif dataset == "imagenet_xception":
        model_name = model_save_dir + "/imagenet_xception_v2.hdf5"
        num_classes = 1000
        num_targeted_classes = 100
    else:
        print("Unknown dataset!")
        exit()

    if args.model_path != 'none':
        model_name = args.model_path

    if args.number_of_target_classes < num_classes and args.number_of_target_classes > 0:
        num_targeted_classes = args.number_of_target_classes

    if dataset == "imagenet_inceptionv3" or dataset == "imagenet_xception":
        gradient_norms_imagenet(dataset, num_classes, num_targeted_classes, num_of_samples_per_class, model_name, gradient_save_dir, args.imagenet_path)
    else:
        gradient_norms(dataset, num_classes, num_targeted_classes, num_of_samples_per_class, model_name, gradient_save_dir)




