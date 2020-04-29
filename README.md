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
$ python train_target_model.py -d cifar_10
```
You choose among 7 dataset/model combinations: mnist, cifar_10, cifar_100, cifar_100_resnet, cifar_100_densenet, imagenet_inceptionv3, imagenet_xception. For ImageNet, it only fetches the pre-trained model from Keras repository for later use. You can change the learning rate, batchsize, and other options. Use help to see all options.

## MI Attack Using Confidence Values
After traing a target model, you can use the following script to launch MI attack based on confidence values:
```
python confidence_based.py -d cifar_10 -a NN -s none
```
The second and third options indicate the MI attack model and sampling, respectively. For MI attack model, you can choose among NN, RF, and XGBoost. For smapling, there are three options: none, undersamping, and oversampling. Try all combinations to get the best result. Use help to see more options.
Note that the MNIST, CIFAR-10, and CIFAR-100 datasets with be fetched from the Keras repository automatically. However, for ImageNet, you need to manually download, and extract the files. Then you need create a folder that contains two subfolders: "Train" and "Test". Then, you need to copy the samples of each ImageNet class in a different folder within the train and test folder. Finally, you need to use the -i option to indicate the ImageNet dataset path.

## MI Attack Using the Output of Intermediate Layers
You can use the following script to launch MI attack based on the output of intermedaite layers:
```
python intermediate_layer.py -d cifar_10 -a NN -s none -l -1
```
Options are similar to the previous attack with one exception. You can choose the layer on which you want to launch the MI attack with -l option. -1 means the layer before the Softmax.


## MI Attack Using Distance to the Decision Boundary
First, you need to run the following script to obtain the distances and store them in a set of files:
```
python obtain_distances.py -d cifar_10 -s 400 -n 5
```
Computing distance to the boundary is a time-consuming process. You can limit the number of samples and number of classes with -s and -n options. -s indicates how many samples per class you want to obtain the distance for and -n indicates how many classes you want to obtain the distance for. After running the script, the distance to the boundary of samples of each class will be stored in a separate file in the "saved_distances" folder.
Next, You need to run the following script to fit a Logistic regression model the distances stored in the files:
```
python fit_model_to_distance.py -d cifar_10
```

## MI Attack Using Gradient Norm
The process is similar to the previous attack. First, you need to obtain the gradients and store them in a set of files by running:
```
python obtain_gradients.py -d cifar_10 -s 400 -n 5
```
Next, You need to run the following script to fit a Logistic regression model the gradients stored in the files:
```
python fit_model_to_gradient.py -d cifar_10
```
