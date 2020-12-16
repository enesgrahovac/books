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
# print(x_input)
# print(y_output)
# print(torch.sum(y_output,dim=1))

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
# print("Mean Squared Error Loss: ",loss)

'''
Categotical Cross-Entropy Loss

typically used in a multiclass classification setting in which
the outputs are interpreted as predictions of class membership probabilities.
'''
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3,5,requires_grad=True)
targets = torch.tensor([1,0,3],dtype=torch.int64)
loss = ce_loss(outputs,targets)
# print("Cross Entropy Loss: ",loss)

'''
Binary Cross-Entropy Loss (BCE loss)

this is better for classifications between 2 classes, aka binary classification.
'''

bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4,1,requires_grad=True))
targets = torch.tensor([1,0,1,0],dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities,targets)
# print(probabilities)
# print("BCE Loss: ", loss)

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

# print(get_toy_data(50))

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
    raw_train_dataset_csv="./project_data/yelp/raw_train.csv",
    raw_test_dataset_csv="./project_data/yelp/raw_test.csv",
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="./project_data/yelp/reviews_with_splits_full.csv",
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

# print(final_reviews.head())

from torch.utils.data import Dataset, DataLoader
import string

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

    def set_split(self, split="train"):
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
        row = self._target_df.iloc[index]

        review_vector = \
            self._vectorizer.vectorize(row.review)

        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.rating)
        
        return {'x_data':review_vector,
                'y_target':rating_index}

    def get_num_batches(self,batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

### In this example, the tokens will be words

### The Vocabulary class below will manage token to integer mapping for the rest of the ML pipeline.

class Vocabulary(object):
    """ Class to process text and extract Vocabulary for mapping """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (book): a flag that indicates whether to add the UNK token
            unk_token (attr): the UNK token to add into the vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token,idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {
            'token_to_idx':self._token_to_idx,
            'add_unk':self._add_unk,
            'unk_token':self._unk_token
        }
    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """ Update mapping dicts based on the token
        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self,token):
        """ Retrieve the index associated with the token
            or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns: 
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
            for the UNK functionality)   
        """
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
        
    def lookup_index(self, index):
        """ Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary".format(index))
        return self._idx_to_token[index]
    
    def __str__(self):
        return "<Vocabulary(size=%d)>".format(len(self))
    
    def __len__(self):
        return len(self._token_to_idx)

""" The next step is to have a Vectorizer that takes the integer form of the
    tokens and outputs a vector of the data points. The Vector should be
    the same length between each input.

    The Vectorizer uses a one-hot approach that returns a vector with the same length as
    the vocab and has a one for each word that is present. It discards the order of the 
    words.
"""

class ReviewVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self,review_vocab, rating_vocab):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
            rating_vocab (Vocabulary): maps class labels to integers
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self,review):
        """ Create a collapsed one-hot vector for the review

        Args:
            review (str): the review
        Returns:
            one_hot (np.ndarray): the collapsed one-hot encoding
        """
        one_hot = np.zeros(len(self.review_vocab),dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
        
        return one_hot
    
    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """ Instantiate the vectorizer from the dataset dataframe

        Args:
            review_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        #Add ratings
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)
        
        # Add top words if count > provided count
        word_counts = collections.Counter() # part of collections library
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)
        
        return cls(review_vocab, rating_vocab)

    def to_serializable(self):
        """ Create the serializable dictionary for caching

        Returns:
            contents (dict): the serializable dictionary
        """
        return {
            'review_vocab': self.review_vocab.to_serializable(),
            'rating_vocab': self.rating_vocab.to_serializable()
        }

"""
    The final step of the text-to-vectorized minibatch pipeline is to 
    group the vectorized data points.

    Grouping into minibatches is a vital part of training neural networks.

    PyTorch provides the DataLoader class for coordinating the process.

"""

def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """ A generator functino which wraps the PyTorch DataLoader. It will ensure
        each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
    
class ReviewClassifier(nn.Module):
    """ A perceptron is one linear layer"""
    def __init__(self, num_features):
        """
        Args:
            input_dim (int): size of the input features
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """ The forward pass of the perceptron
        Args: 
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
            Returns:
                the resulting tensor. tensor.shape should be (batch,).
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out # the activation function used here is the sigmoid function

""" The Training Routine

    Outline of the components of the training routine and how they come together
    with the dataset and model to adjust the model parameters and increase its performance.
"""

from argparse import Namespace

args = Namespace(
    #Data and path information
    frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='project_data/yelp/reviews_with_splits_lite.csv',
    save_dir='model_storage/ch3/yelp/',
    vectorizer_file='vectorizer.json',
    # No Model Hyperparameters
    # Training hyperparameters
    batch_size = 128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # Runtime options omitted for space
)

### Instantiating the dataset, model, loss, optimizer, and training state
import torch.optim as optim

def make_train_state(args):
    return {
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

# dataset and vectorizer
dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
vectorizer = dataset.get_vectorizer()

# model
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
classifier = classifier.to(args.device)

# loss and optimizer
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

### The Training Loop
for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index

    # Iterate over training dataset

    # setup: batch generator, set loss and acc to 0, set train mode on
    dataset.set_split('train')
    batch_generator = generate_batches(dataset,batch_size=args.batch_size,device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()

    for batch_index, batch_dict in enumerate(batch_generator):
        # The training routine in 5 steps:

        # step 1. zero the gradients
        optimizer.zero_grad()

        # step 2. compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # step 3. compute the output
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # step 4. use loss to produce gradients
        loss.backward()

        # step 5. use optimizer to take gradient step
        optimizer.step()

        # ------------------------------------
        # Compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)
    
    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    #Iterate over val dataset

    # setup: batch generator, set loss and acc to -, set eval mode on
    dataset.set_split('val')
    batch_generator = generate_batches(dataset, batch_size=args.batch_size,device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):

        #step 1. Compute the output
        y_pred = classifier(x_in=batch_dict["x_data"].float())

        # Step 2. compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # Step 3. compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index  + 1)

    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)

### Test the model
dataset.set_split('test')
batch_generator = generate_batches(dataset,batch_size=args.batch_size,device=args.device)
running_loss = 0.0
running_acc = 0.0
classifier.train()

for batch_index, batch_dict in enumerate(batch_generator):

    # compute the output
    y_pred = classifier(x_in=batch_dict['x_data'].float())

    # compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'].float())
    loss_batch = loss.item()
    running_loss += (loss_batch - running_loss) / (batch_index + 1)

    # Compute the accuracy
    acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_batch - running_acc) / (batch_index + 1)

train_state['test_loss'].append(running_loss)
train_state['test_acc'].append(running_acc)

print(train_state['test_loss'])
print(train_state['test_acc'])

