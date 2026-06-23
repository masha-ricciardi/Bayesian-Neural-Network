import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Importing MNIST 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Brightness of pixel values goes from 0-255, we can divide by 255 to normalise and make our scale 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convolutional layers expect 3 dimensions, since our imported images have only height and width,
#we add a 3rd dimension for 1 colour channel (grayscale).
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


"""
Here we define our neural network model. 
First we create a sequential model, allowing us to stack layers on top of each other.
The first layer is a convolutional layer with 16 filters (feature detectors), each of which has a 3x3 kernel, and ReLU activation.
ReLU (Rectified Linear Unit) is a common activation function that introduces non-linearity to the model, allowing it to learn complex patterns.
It's computationally inexpensive and helps mitigate the vanishing gradient problem since the gradient is either 0 for negative inputs or 1 for positive inputs.

The next layer is a max pooling layer, it reduces the spatial dimensions of the output from the previous layer by taking the maximum value in each 2x2 window.
This helps reduce the number of parameters and computations in the network and makes it more robust against small translations in the input.

The 3rd layer is another convolutional layer, exactly like the first one but with 32 filters.
This layer learns more complex features from the pooled output of the previous layer.
The 4th layer is another max pooling layer, again reducing the spatial dimensions of the output.
Layer 5 flattens the output from the previous layer from a 2D array into a 1D array, this prepares the data for the upcoming layer.
Layer 6 is a fully connected layer with 128 neurons and ReLU activation, this adds non-linearity and allows the model to learn complex patterns.
The final layer is a final output layer with 10 neurons (one for each digit 0-9).
It uses softmax activation, which converts the output into probabilities that sum to 1, allowing us to interpret the output as a probability distribution over the 10 classes.
"""

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# This summarises each layer of the model, showing the output shape and number of parameters for each layer.
model.summary()

# Model.compile prepares the model for training. It specifies the optimizer (ADAM), the loss function (sparse categorical crossentropy),
# and the metrics to monitor during training (accuracy).
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Next we train the model using the training data. We specify the number of epochs (5) and the batch size (128).
model.fit(x_train, y_train, epochs=5, batch_size=128)

# We evaluate the model on the test data to see how well it performs on unseen data.
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

'''
The model is now trained and evaluated. The test accuracy (0.9854000210762024) gives us an indication of how well the model generalizes to new, unseen data.
The test loss (0.044579893350601196) indicates the average error of the model on the test data.
'''