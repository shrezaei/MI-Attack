import argparse
import os
from attacks.confidence_based_attack import conf_based_attack
from attacks.confidence_based_attack_imagenet import conf_based_attack_imagenet

parser = argparse.ArgumentParser(description='MI attack besed on confidence values.')
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'], help='Indicate dataset and target model. If you trained your own target model, the model choice will be overwritten')
parser.add_argument('-m', '--model_path', type=str, default='keras_models/cifar10/alexnet.h5', help='Indicate the path to the target model. If you used the train_target_model.py to train the model, leave this field to the default value.')
parser.add_argument('-a', '--attack_model', type=str, default='NN', choices=['NN', 'RF', 'XGBoost'], help='MI Attack model (default is NN).')
parser.add_argument('-s', '--sampling', type=str, default='none', choices=['none', 'undersampling', 'oversampling'], help='Indicate sampling. Useful for highly imbalaned cases.')
parser.add_argument('-c', '--attacker_knowledge', type=float, default=0.8, help='The portion of samples available to the attacker. Default is 0.8.')
parser.add_argument('-n', '--number_of_target_classes', type=int, default=0, help='Limit the MI attack to limited a number of classes for efficiency!')
parser.add_argument('-i', '--imagenet_path', type=str, default='../imagenet/', help='Path to the imagenet dataset if the attack is on Imagenet.')
parser.add_argument('--save_confidence_histograms', default=False, help='Save confidence histogram of each class.', action='store_true')
parser.add_argument('--show_separate_results', default=False, help='Show results for correctly classified and misclassified samples, separately.', action='store_true')
parser.add_argument('--verbose', default=False, help='Print full details.', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    verbose = args.verbose
    dataset = args.dataset
    model_name = args.model_path
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
    report_separated_performance = args.show_separate_results

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    num_classes = 10
    if dataset == "mnist" or dataset == "cifar10":
        num_classes = 10
        num_targeted_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
        num_targeted_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000
        num_targeted_classes = 100
    else:
        print("Unknown dataset!")
        exit()

    if num_classes > args.number_of_target_classes > 0:
        num_targeted_classes = args.number_of_target_classes

    if dataset == "imagenet":
        conf_based_attack_imagenet(dataset, attack_classifier, sampling, what_portion_of_samples_attacker_knows, save_confidence_histogram, report_separated_performance, num_classes, num_targeted_classes, model_name, verbose, args.imagenet_path)
    else:
        conf_based_attack(dataset, attack_classifier, sampling, what_portion_of_samples_attacker_knows, save_confidence_histogram, report_separated_performance, num_classes, num_targeted_classes, model_name, verbose)
