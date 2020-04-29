# On the Infeasibility of Membership Inference on Deep Models
The paper is available at: www.archix,org ...

## Library Versions
* Python 3.5
* Keras v2.2.5
* Tensorflow v1.12.0
* Warning: Theano backend is not tested for now.

## Train a Target Model
You lunch an attack, you first need to train a model. You can train a target model by running:
```
$ python train_target_model.py -d cifar_100
```
You choose among 7 dataset/model combinations: mnist, cifar_10, cifar_100, cifar_100_resnet, cifar_100_densenet, imagenet_inceptionv3, imagenet_xception. For ImageNet, it only fetches the pre-trained model from Keras repository for later use. You can change the learning rate, batchsize, and other options. Use help to see all options.

## MI Attack Using Confidence values
