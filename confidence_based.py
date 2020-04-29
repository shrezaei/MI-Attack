import argparse
import os
from attacks.confidence_based_attack import conf_based_attack
from attacks.confidence_based_attack_imagenet import conf_based_attack_imagenet

parser = argparse.ArgumentParser(description='MI attack besed on confidence values.')
parser.add_argument('-d', '--dataset', type=str, default='cifar_10', choices=['mnist', 'cifar_10', 'cifar_100', 'cifar_100_resnet', 'cifar_100_densenet', 'imagenet_inceptionv3', 'imagenet_xception'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-m', '--model_path', type=str, default='none', help='Indicate the path to the target model. If you used the train_target_model.py to train the model, leave this field to the default value.')
parser.add_argument('-a', '--attack_model', type=str, default='NN', choices=['NN', 'RF', 'XGBoost'], help='MI Attack model (default is NN).')
parser.add_argument('-s', '--sampling', type=str, default='none', choices=['none', 'undersampling', 'oversampling'], help='Indicate sampling. Useful for highly imbalaned cases.')
parser.add_argument('-c', '--attacker_knowledge', type=float, default=0.8, help='The portion of samples available to the attacker. Default is 0.8.')
parser.add_argument('-n', '--number_of_target_classes', type=int, default=0, help='Limit the MI attack to limited a number of classes for efficiency!')
parser.add_argument('-i', '--imagenet_path', type=str, default='../imagenet/', help='path to the imagenet dataset.')
parser.add_argument('--save_confidence_histograms', default=False, help='Save confidence histogram of each class.', action='store_true')
parser.add_argument('--train_for_correct_only', default=False, help='Train separate attack models for correctly labeled samples.', action='store_true')
parser.add_argument('--verbose', default=False, help='Print full details.', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    verbose = args.verbose
    dataset = args.dataset
    attack_classifier = args.attack_model
    sampling = args.sampling
    what_portion_of_samples_attacker_knows = args.attacker_knowledge
    if what_portion_of_samples_attacker_knows < 0.1 or what_portion_of_samples_attacker_knows > 0.9:
        print('Error: Attacker knowledge should be in [0.1, 0.9] range!')
        exit()

    save_confidence_histogram = args.save_confidence_histograms
    if save_confidence_histogram:
        save_histogram_dir = os.path.join(os.getcwd(), 'figures/conf histogram/' + dataset)
        if not os.path.isdir(save_histogram_dir):
            os.makedirs(save_histogram_dir)
    show_MI_attack_separate_result = args.train_for_correct_only

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
        conf_based_attack_imagenet(dataset, attack_classifier, sampling, what_portion_of_samples_attacker_knows, save_confidence_histogram, show_MI_attack_separate_result, num_classes, num_targeted_classes, model_name, verbose, args.imagenet_path)
    else:
        conf_based_attack(dataset, attack_classifier, sampling, what_portion_of_samples_attacker_knows, save_confidence_histogram, show_MI_attack_separate_result, num_classes, num_targeted_classes, model_name, verbose)
