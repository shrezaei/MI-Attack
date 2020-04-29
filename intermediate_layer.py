import argparse
import os
from attacks.intermediate_layer_attack import intermediate_layer_attack
from attacks.intermediate_layer_attack_imagenet import intermediate_layer_attack_imagenet

parser = argparse.ArgumentParser(description='MI attack besed on intermediate layers output.')
parser.add_argument('-d', '--dataset', type=str, default='cifar_10', choices=['mnist', 'cifar_10', 'cifar_100', 'cifar_100_resnet', 'cifar_100_densenet', 'imagenet_inceptionv3', 'imagenet_xception'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-m', '--model_path', type=str, default='none', help='Indicate the path to the target model. If you used the train_target_model.py to train the model, leave this field to the default value.')
parser.add_argument('-a', '--attack_model', type=str, default='NN', choices=['NN', 'RF', 'XGBoost'], help='MI Attack model (default is NN).')
parser.add_argument('-s', '--sampling', type=str, default='none', choices=['none', 'undersampling', 'oversampling'], help='Indicate sampling. Useful for highly imbalaned cases.')
parser.add_argument('-c', '--attacker_knowledge', type=float, default=0.8, help='The portion of samples available to the attacker. Default is 0.8.')
parser.add_argument('-n', '--number_of_target_classes', type=int, default=0, help='Limit the MI attack to limited a number of classes for efficiency!')
parser.add_argument('-i', '--imagenet_path', type=str, default='../imagenet/', help='path to the imagenet dataset.')
parser.add_argument('-l', '--intermediate_layer', type=int, default=-1, help='Possible values: {-1, -2, -3}. May varies based on the target model')
parser.add_argument('--no_train_for_all', default=True, help='Disable training an attack model for all samples.', action='store_false')
parser.add_argument('--no_train_for_correctly_classified', default=True, help='Disable training a separate attack model for correctly labeled samples.', action='store_false')
parser.add_argument('--no_train_for_incorrect_misclassified', default=True, help='Disable training a separate attack model for misclassifed labeled samples.', action='store_false')
parser.add_argument('--verbose', default=False, help='Print full details.', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    verbose = args.verbose
    dataset = args.dataset
    intermediate_layer = args.intermediate_layer
    attack_classifier = args.attack_model
    sampling = args.sampling
    what_portion_of_samples_attacker_knows = args.attacker_knowledge
    if what_portion_of_samples_attacker_knows < 0.1 or what_portion_of_samples_attacker_knows > 0.9:
        print('Error: Attacker knowledge should be in [0.1, 0.9] range!')
        exit()

    show_MI_attack = args.no_train_for_all
    show_MI_attack_separate_result = args.no_train_for_correctly_classified
    show_MI_attack_separate_result_for_incorrect = args.no_train_for_incorrect_misclassified

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if dataset == "mnist" or dataset == "cifar_10":
        model_name = save_dir + '/' + dataset + '_weights_' + 'final.h5'
        num_classes = 10
        num_targeted_classes = 10
    elif dataset == "cifar_100" or dataset == "cifar_100_resnet" or dataset == "cifar_100_densenet":
        model_name = save_dir + '/' + dataset + '_weights_' + 'final.h5'
        num_classes = 100
        num_targeted_classes = 100
    elif dataset == "imagenet_inceptionv3":
        model_name = save_dir + "/imagenet_inceptionV3_v2.hdf5"
        num_classes = 1000
        num_targeted_classes = 100
    elif dataset == "imagenet_xception":
        model_name = save_dir + "/imagenet_xception_v2.hdf5"
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
        intermediate_layer_attack_imagenet(dataset, intermediate_layer, attack_classifier, sampling, what_portion_of_samples_attacker_knows, num_classes, num_targeted_classes, model_name, verbose, show_MI_attack, show_MI_attack_separate_result, show_MI_attack_separate_result_for_incorrect, args.imagenet_path)
    else:
        intermediate_layer_attack(dataset, intermediate_layer, attack_classifier, sampling, what_portion_of_samples_attacker_knows, num_classes, num_targeted_classes, model_name, verbose, show_MI_attack, show_MI_attack_separate_result, show_MI_attack_separate_result_for_incorrect)
