'''
Foundational Components of Neural Networks
'''

'''
The simplest neural network unit is a perceptron. It is modelled after the biological neuron.

y=f(wx+b) , where the 3 knobs are the set of weights (w), a bias (b), and an activation 
function f. y is the output and x is the input.

'''

### Implementing a perceptron using PyTorch

import torch
import torch.nn as nn

class Perceptron(nn.Module):
    """ A perceptron is one linear layer"""
    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """ The forward pass of the perceptron
        Args: 
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
            Returns:
                the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze() # the activation function used here is the sigmoid function

### Below are 4 important activation functions: sigmoid, tanh, ReLU, and softmax

'''
The sigmoid activation function is one of the earliest and keeps a range between 0,1.

The sigmoid is represented by f(x) = 1/(1+exp(-x))

below is a graph of what it looks like.
'''

import matplotlib.pyplot as plt

x = torch.arange(-5,5,0.1)
y = torch.sigmoid(x)
# plt.plot(x.numpy(),y.numpy())
# plt.show()

'''
The Tanh activation function is a cosmetically different variant of the sigmoid.

f(x) = tanh(x) = ((exp(x)-exp(-x))/(exp(x)+exp(-x)))

below shows the tanh function.

The range is from [-1,1]
'''

y = torch.tanh(x)
# plt.plot(x.numpy(),y.numpy())
# plt.show()

''' 
ReLU (ray-luh) stands for rectified linear unit.

"the most important of activation functions"

f(x) = max(0,x)

'''

relu = torch.nn.ReLU()
y =relu(x)
# plt.plot(x.numpy(),y.numpy())
# plt.show()

'''
Parametric ReLU (PReLU) activations functions have been
proposed to mitigated the negative clipping effects of 
ReLU where many outputs in the network can simply
become 0 and never revive again.

f(x) = max(x,ax)
'''

prelu = torch.nn.PReLU(num_parameters=1)
y = prelu(x)
# plt.plot(x.numpy(),y.detach().numpy())
# plt.show()

'''
Softmax gives a discrete probability distribution over k possible classes.
'''

softmax = nn.Softmax(dim=1)
x_input = torch.randn(1,3)
y_output = softmax(x_input)
print(x_input)
print(y_output)
print(torch.sum(y_output,dim=1))

### Below are commonly used Loss Functions

'''
A loss function takes a truth (y) and a prediction (y_hat)
as an input and calculates a real-valued score.
'''

'''
Mean Squared Error Loss (mse)
It is common for regression problems.

Mean absolute error (MAE) and root mean squared error (RMSE) are also common.
'''

mse_loss = nn.MSELoss()
outputs = torch.randn(3,5, requires_grad=True)
targets = torch.randn(3,5)
loss = mse_loss(outputs, targets)
print("Mean Squared Error Loss: ",loss)

'''
Categotical Cross-Entropy Loss

typically used in a multiclass classification setting in which
the outputs are interpreted as predictions of class membership probabilities.
'''
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3,5,requires_grad=True)
targets = torch.tensor([1,0,3],dtype=torch.int64)
loss = ce_loss(outputs,targets)
print("Cross Entropy Loss: ",loss)

'''
Binary Cross-Entropy Loss (BCE loss)

this is better for classifications between 2 classes, aka binary classification.
'''

bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4,1,requires_grad=True))
targets = torch.tensor([1,0,1,0],dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities,targets)
print(probabilities)
print("BCE Loss: ", loss)