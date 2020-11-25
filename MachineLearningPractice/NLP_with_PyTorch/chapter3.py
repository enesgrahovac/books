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

'''
The following example is a toy problem: classifying two-dimensional points
into one of two classes. This means learning a single line or plane, called
a decision boundary or hyperplane.
'''

# Constructing Toy Data 
import numpy as np
LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)
def get_toy_data(batch_size, left_center=LEFT_CENTER, right_center=RIGHT_CENTER):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)

print(get_toy_data(50))

# Choosing the Model: Perceptron, for it allows for any input size.

# Choosing the Loss Function: BCE for there are binary classes: stars and circles

# Choosing an optimizer: the hyperparameter is called a learning rate, which controls how much
# impact the error signla has on updating the weights. PyTorch has many choices for 
# optimizers. Stochastic Gradient Descent (SGD), is common, but has convergence issues.
# The current preferred alternative are adaptive optimizers, such as Adagrad or Adam

# Instatiating the Adam Optimizer
import torch.optim as optim

def deep_learning_perceptron():
    input_dim = 2
    lr = 0.001

    perceptron = Perceptron(input_dim=input_dim)
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

    batch_size = 1000
    n_epochs = 12
    n_batches = 5

    # each epoch is a complete pass over the training data
    for epoch_i in range(n_epochs):
        # the inner loop is over the batches in the dataset
        for batch_i in range(n_batches):

            #step 0: Get the data
            x_data, y_target = get_toy_data(batch_size)

            #step 1: Clear the gradients
            perceptron.zero_grad()

            #step 2: Compute the forward pass of the model
            y_pred = perceptron(
                x_data,
                # apply_sigmoid=True
                )

            #step 3: Compute the loss value that we wish to optimize
            loss = bce_loss(y_pred,y_target)

            #step 4: Propogate the loss signal backward
            loss.backward()

            #step 5: Trigger the optimizer to perform one update
            optimizer.step()

'''
The following is a machine learning example of guessing a restaurants rating
based on reviews alone. So, is it a good or badly rated restaurant solely
based on what people reviewed it to be?
'''

import collections
from argparse import Namespace
import pandas as pd
import re

# Load Data
args = Namespace(
    raw_train_dataset_csv="yelp/raw_train.csv",
    raw_test_dataset_csv="yelp/raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="yelp/reviews_with_splits_full.csv",
    seed=1337
)
train_reviews = pd.read_csv(args.raw_train_dataset_csv, header=None, names=['rating', 'review'])
train_reviews = train_reviews[~pd.isnull(train_reviews.review)]
test_reviews = pd.read_csv(args.raw_test_dataset_csv, header=None, names=['rating', 'review'])
test_reviews = test_reviews[~pd.isnull(test_reviews.review)]

# Step one is to divide the data into training, validation, and testing splits
by_rating = collections.defaultdict(list)
for _, row in train_reviews.iterrows():
    by_rating[row.rating].append(row.to_dict())

final_list = []
np.random.seed(args.seed)

for _, item_list in sorted(by_rating.items()):
    np.random.shuffle(item_list)

    n_total = len(item_list)
    n_train = int(args.train_proportion * n_total)
    n_val = int(args.val_proportion * n_total)
    n_test = int(args.test_proportion * n_total)

    # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'

    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    
    for item in item_list[n_train+n_val:n_train+n_val+n_test]:
        item['split'] = 'test'
    
    # Add to final list
    final_list.extend(item_list)

final_reviews = pd.DataFrame(final_list)

### Minimally cleaning the data

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])",r" \1", text)
    text = re.sub(r"[^a-zA-Z.,!?]+",r" ", text)
    return text

final_reviews.review = final_reviews.review.apply(preprocess_text)

print(final_reviews.head())

from torch.utils.data import Dataset

### A PyTorch Dataset class for the Yelp Review Dataset

class ReviewDataset(Dataset):
    def __init__(self,review_df,vectorizer):
        """
        Args:
            review_df (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split=='val']
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df,self.train_size),
            'val': (self.val_df,self.val_size),
            'test': (self.test_df,self.test_size)
        }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """
        Load dataset and make a new vectorizer from scratch

        Args:
            review_csv (str): location of the dataset
        Returns:
            an instance of ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df,ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train")
        """ selects the splits in the dataset using a solumn in the dataframe

        Args:
            split (str): one of "train","val", or "test"
        """

        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self,index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dict of the data point's features (x_data) and label (y_targets)
        """
        row = self.__target_df.iloc[index]

        review_vector = \
            self._vectorizer.vectorize(row.review)

        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.rating)
        
        return ('x_data':review_vector,
                'y_target':rating_index)

    def get_num_batches(self,batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
    
