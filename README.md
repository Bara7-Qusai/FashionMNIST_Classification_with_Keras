### Neural Networks

Neural networks are computational models inspired by the human brain. They consist of units called "neurons" (or "nodes") connected to each other. Neural networks can learn patterns from data by adjusting the weights and biases between these neurons. Deep neural networks (DNNs) are a type of neural network that contains multiple hidden layers between the input and output layers.

### How Neural Networks Work

1. **Input Layer**: This layer receives the initial data (such as images, texts, etc.).
2. **Hidden Layers**: These layers process the data by applying mathematical operations (called transformations). The data is represented in new dimensions at each hidden layer.
3. **Output Layer**: This layer provides the final result (such as classifying an image into a shirt, trousers, etc.).

Each neuron in one layer is connected to neurons in the next layer through a weight. Biases are also added to modify the output values.

#### The Anatomy of a Neural Network

- **Neurons**: The basic units of a neural network. Each neuron holds a single value and passes it through an activation function.
- **Weights**: These are the parameters that connect neurons between layers. They determine the strength of the connection.
- **Biases**: Additional parameters added to each layer to shift the activation function, providing more flexibility.
- **Activation Functions**: Functions applied to the weighted sum of inputs for a neuron. Common activation functions include ReLU (Rectified Linear Unit), Tanh (Hyperbolic Tangent), and Sigmoid.

### How Information is Processed

When data is fed into the input layer, it gets transformed as it passes through each layer. At each neuron, the data undergoes the following transformation:

\[ Y = F\left(\left(\sum_{i=0}^{n} w_i x_i\right) + b\right) \]

Where:
- \( w \) represents the weights.
- \( x \) represents the input values.
- \( b \) represents the bias.
- \( F \) is the activation function.

The network starts with random weights and biases. During training, it adjusts these parameters to minimize the difference between the predicted output and the actual output using a technique called backpropagation.

### Training the Neural Network

**Backpropagation**: This is the fundamental algorithm behind training neural networks. It involves:
- **Loss/Cost Function**: Measures how well the network's predictions match the actual results. Common loss functions include Mean Squared Error (MSE) and Cross-Entropy Loss.
- **Gradient Descent**: An optimization algorithm that adjusts the weights and biases to minimize the loss function. It involves calculating the gradient of the loss function and moving in the direction of steepest descent.

### Code Example: Simple Neural Network with Keras

We'll create a simple neural network to classify images of clothing using the Keras library.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
    keras.layers.Dense(128, activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax') # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy:', test_acc)

# Make predictions
predictions = model.predict(test_images)

# Function to plot results
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# Display the result for a specific image
num = 0  # Choose the image number
plt.figure()
plot_image(num, predictions, test_labels, test_images)
plt.show()
```

### Code Explanation

1. **Loading the Data**: We load the `Fashion MNIST` dataset from Keras, which includes images of various clothing items.
2. **Scaling the Data**: We scale the pixel values to be between 0 and 1 to make them easier to process.
3. **Building the Model**: We create a sequential model with an input layer (flattened), one hidden layer, and an output layer.
4. **Compiling the Model**: We specify the optimizer (Adam), the loss function (Sparse Categorical Crossentropy), and the metric (accuracy).
5. **Training the Model**: We train the model on the training data for 10 epochs.
6. **Evaluating the Model**: We test the model on the test data and print the accuracy.
7. **Making Predictions**: We use the model to make predictions on the test images.
8. **Displaying Results**: We plot the predicted label and the actual label for a specific image.

This code provides a basic introduction to building, training, and evaluating a neural network using Keras.
