# Artificial Neural Network

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
