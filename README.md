# Artificial Neural Network
Machine learning is divided in 3 big groups _Supervised Learning_, _Unsupervised Learing_ and _Reinforcement Learning_:
- _Supervised Learning_: Has 2 subgroups:
  - **Classification**: Binary outputs, is related to -> Identity fraud detection, Image classification, Customer Retention and Diagnostics.
  - **Regression**: Real values (%) -> Advertising popularity prediction, Weather forecasting, Market forecasting, Estimating life expectancy, Population growth prediction.
- _Unsupervised Learning_: Has 2 subgroups:
  - **Dimensionality reduction**: Feature elicitation, Structure discovery, Meaningful compression, Big data visualisation.
  - **Clustering** Customer segmentation, Targeted marketing, Recommender systems.
- _Reinforcement Learning_: Game AI, Skill Acquisition, Learning tasks, Robot Navigation, Real-time decisions.

> NOTE: In this case I am going to focus on Supersvised learnig and more specifically on Classification ML

## Arquitectures
### Introduction example 
![Feed](https://github.com/RogerCL24/ANN/assets/90930371/1f27fc5b-89c6-4f0e-b66b-4354a14926aa)

[Source](https://learnopencv.com/understanding-feedforward-neural-networks/)

Here we got a Neural Network with 3 layers each layers has 1 or more ``perceptrons``, the perceptrons from the input layer send the data to the hidden layer and finally to the output layer, is a ``feed-forward`` network, it goes from layer to layer until the end

### Arquitecture types
- **Feedforward Neural Network**: Is the first and simplest neural network created, the processed data goes only forward and not in cycle like the majority of neural networks.
- **Convolutional Neural Network or CNN**: Mostly applied in the image processing related field, the perceptrons are receptive because they work as biological human eyes.
![Convo](https://github.com/RogerCL24/ANN/assets/90930371/2479b38c-fcc4-494c-970d-c584f86279c1)

[Source](https://www.jessicayung.com/explaining-tensorflow-code-for-a-convolutional-neural-network/)
- **Recurrent Neural Network or RNN**: It works like a cycle because the hidden layer has feedback, its used in the language field to create paragraphs or sentences with the context.
![Recurrent](https://github.com/RogerCL24/ANN/assets/90930371/6cd5c133-c2ef-4eda-bf17-e0ea6c1a5b4b)

[Source](https://zhuanlan.zhihu.com/p/37290775)
- **Radial basis function Neural Network or RBF**: It's like the ``feedforward`` neural network but in the hidden layer the output of the function is calculated depending on the distance the center is.
- **Modular Neural Network**: Like the name describes, there are isolated neural networks which compose the Modular neural network, namely, each isolated neural networks is a module.

## Structure

1. **IMPORT**: Import the needed libraries like tensorflow(keras), numpy, matplotlib -> First lines of [1_FirstDataSet.ipynb](https://github.com/RogerCL24/ANN/blob/main/1_FirstDataSet.ipynb)
2. **DATA**: What data are going to use to train our model, and which part is will be train data and which test data <sub> train data should be 80%/70% and test data 20%/30% </sub> -> Machine Learning section at [1_FirstDataSet.ipynb](https://github.com/RogerCL24/ANN/blob/main/1_FirstDataSet.ipynb)
3. **MODEL**: Arquitecture of the model:
  - Model layers: Input layers, hidden layers, output layers, if they are going to be dense layers, types of activation functions (sigmoid, tanh, softmax, ReLU...).
  - Model compilation: Optimization, error comparisons, data transfer between layers. 
4. **TRAIN(fit) & EVALUATE(predict)**: Training the model with the treated dataset (train data) several times and then evaluate the model with new data (test data) to verify the performance. -> Example of this and `MODEL` at [3_TensorFlow.ipynb](https://github.com/RogerCL24/ANN/blob/main/3_TensorFlow.ipynb)

## Model
### Model layers (activation functions) <sub> Most used </sub>
- **ReLU**: Any data input will be positive, if the input is negative then the output is 0 otherwise it has the same value.

![ReLU](https://github.com/RogerCL24/ANN/assets/90930371/9ddb3419-696b-492d-aecb-b71f7b5ea59d)

- **Sigmoid**: Used to classify binary, if there are more classification possibilities than 1 or 0 we use a Sigmoid derivative.
  
![SIgmoid](https://github.com/RogerCL24/ANN/assets/90930371/5d8f21df-2172-47c8-9c85-5aba77f035de)

- **tanh**: hiperbolic tangent.

![tanh](https://github.com/RogerCL24/ANN/assets/90930371/72f7ca53-6d9a-4853-87e9-28c2c5988555)

<sub> Image source from the EDteam organization </sub>

### Model compiling (Optimizers)
There are sevel types of optimizers like AdaGrad, RMSProp, SGDNesterov, AdaDelta, Adam... used to reduce the training cost, for example in MNIST data set (images of the 0 to 9 digits)

![Adam](https://github.com/RogerCL24/ANN/assets/90930371/2f248f59-8f17-4e8e-a5b8-23dbe5f0cdc5)

[Source](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning)

Adam is the best optimization algorithm in this case, with the same amount of iterations over the dataset has less training cost.

> OBSERVATION: All the optimization algorithms and activation functions are already implemented at the TensorFlow(keras) library

### Example of the model functionality with the MNIST dataset
![image](https://github.com/RogerCL24/ANN/assets/90930371/f03d5f38-f9c4-4830-90a5-bc42650c0770)

## Performance assessment
In order to check if our model is learning properly or not, we need to look for overfitting or underfitting errors, comparing the real value and the predicted one....This way we know what we have to do, increase the amount of data to train, decrease it, change the model alternating layers, use a diferent optmizators algorithms....The main goal is get the lowest error.

On _Supervised Learning_: <sub> We are going to focus only in this one </sub>
- **Classification** techniques: Classification Accuracy, Confusion Matrix, Area under curve (AUC)...
- **Regression** techniques: Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) <sub> There are more </sub>
### Confusion Matrix 
![image](https://github.com/RogerCL24/ANN/assets/90930371/912019bc-2721-4c11-a55f-cae838e36e6c)

Imagine you have several animal pictures and a trained model created to detect dogs, if it's a dog will be a positive otherwise a negative:
- The first photo is a dog therefore Reference/Real value is positive, if the model get it right the predicted value is positive, that means is a True Positive (TP) otherwise would be a False Negative (FN)
- The second photo is a cat therefore Reference/Real value is negative, if the model get it right the predicted value is negative, that means is a True Negative(TN) otherwise would be a False Positive (FP)
![image](https://github.com/RogerCL24/ANN/assets/90930371/3d767f26-ac6d-4168-a0ee-0e8859c3fb2e)

[Source](https://www.researchgate.net/figure/Calculation-of-Precision-Recall-and-Accuracy-in-the-confusion-matrix_fig3_336402347)

To evaluate the performance we use this functions, how many were classified correctly and how many were classifeid wrong.


