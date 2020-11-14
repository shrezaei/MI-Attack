# On the Difficulty of Membership Inference Attacks
The paper is available at: www.archix,org ...

## Library Versions
* Python 3.5
* Keras v2.2.4
* Tensorflow v1.12.2
* Warning: Theano backend is not tested for now.
In case you want to train or convert pytorch models (from bearpaw repository):
* Torch 1.1.0
* pytorch2keras


## Train a Target Model
You lunch an attack, you first need to train a model. You can use any model you have previously trained with Keras/Tensorflow. To skip this part, you can directly download the models we used from https://drive.google.com/file/d/1XQBo3rlZO58mSK57T2Rmv-kniVcvSe9g/view?usp=sharing.
To train an mnist model:
```
$ python train_target_model.py -d cifar_10
```
To load ImageNet models from Keras repository (InceptionV3 or Xception):
```
$ python train_target_model.py -d imagenet_inceptionv3
$ python train_target_model.py -d imagenet_xception
```
For Cifar10 and Cifar100, we convert the models trained by bearpaw (see the link the bearpaw/link.txt). After training a model with bearpaw's codes, you can convert them as follows as an example:
```
$ python torch_to_keras.py -d cifar10 -m resnet-110 -t 'torch_models/cifar10/resnet-110.pth.tar' -k 'keras_models/cifar10/resnet-110.h5'
```


## MI Attack Using Confidence Values
After traing a target model, you can use the following script to launch MI attack based on confidence values:
```
$ python confidence_based.py -d cifar10 -s none -a NN -m keras_models/cifar10/alexnet.h5
```
The second option indicates the dataset. The third option indicates if you want to do undersampling, oversampling, or none to tackle imbalancedness issue. For MI attack model, you can choose among NN, RF, and XGBoost. For smapling, there are three options: none, undersamping, and oversampling. Try all combinations to get the best result. Use help to see more options. The last option indicates the path of the target victim model.
Note that the MNIST, CIFAR-10, and CIFAR-100 datasets with be fetched from the Keras repository automatically. However, for ImageNet, you need to manually download, and extract the files. Then you need to create a folder that contains two subfolders: "Train" and "Test". Then, you need to copy the samples of each ImageNet class in a different folder within the train and test folder. Finally, you need to use the -i option to indicate the ImageNet dataset path.


## MI Attack Using the Output of Intermediate Layers
Similarly, you can use the following script to launch MI attack based on the output of intermedaite layers:
```
$ python intermediate_layer.py -d cifar10 -a NN -s none -l -1 -m keras_models/cifar10/alexnet.h5
```
Options are similar to the previous attack with one exception. You can choose the layer on which you want to launch the MI attack with -l option. -1 means the layer before the Softmax.


## MI Attack Using Distance to the Decision Boundary
First, you need to run the following script to obtain the distances and store them in a set of files:
```
$ python obtain_distances.py -d cifar10 -s 400 -n 10 -m keras_models/cifar10/alexnet.h5 -p saved_distances/cifar10/alexnet
```
Computing distance to the boundary is a time-consuming process. You can limit the number of samples and number of classes with -s and -n options. -s indicates how many samples per class you want to obtain the distance for and -n indicates how many classes you want to obtain the distance for. After running the script, the distance to the boundary of samples of each class will be stored in separate files in the "saved_distances" folder.
Next, You need to run the following script to fit a Logistic regression model the distances stored in the files:
```
$ python fit_model_to_distance.py -d cifar10 -p saved_distances/cifar10/alexnet
```

## MI Attack Using Gradient Norm
The process is similar to the previous attack. First, you need to obtain the gradients and store them in a set of files by running:
```
$ python obtain_gradients.py -d cifar10 -s 400 -n 10 -m keras_models/cifar10/alexnet.h5 -p saved_gradients/cifar10/alexnet
```
Next, You need to run the following script to fit a Logistic regression model the gradients stored in the files:
```
$ python fit_model_to_gradient.py -d cifar10 -p saved_gradients/cifar10/alexnet
```
