import torch

def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

describe(torch.Tensor(2,3)) ## Creates a Tensor with size [2,3]
describe(torch.rand(2,3)) ## Uniform Normal
describe(torch.randn(2,3)) ## Random Normal

### example 1-5
describe(torch.zeros(2,3))
x = torch.ones(2,3)
describe(x)
x.fill_(5)  ## When a PyTorch method has an underscore, it refers to an in-place operation; 
            ## it modifies the content in place without creating a new object
describe(x)
x.fill_(.5)
describe(x)

### Creating a Tensor from a list
x = torch.Tensor([[1,2,3],[4,5,6]])
describe(x)

### Initializing and creating a tensor from a numpy array
import numpy as np
npy = np.random.rand(2,3)
describe(torch.from_numpy(npy))

### Changing the type of the tensor
x = torch.FloatTensor([[1,2,3],[4,5,6]])
describe(x)

x = x.long()
describe(x)

x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.int64)
describe(x)

x = x.float()
describe(x)

### Addition

describe(torch.add(x,x))
describe(x+x)

### Altering dimensions of tensors

x = torch.arange(6)
describe(x)

x = x.view(2,3) ## Rearranges the tensor into a new shape
describe(x)

describe(torch.sum(x,dim=0)) ## Sums columns of tensor
describe(torch.sum(x,dim=1)) ## Adds rows of tensor

describe(torch.transpose(x,0,1))

### Slicing and Indexing a Tensor

x = torch.arange(6).view(2,3)
describe(x)
describe(x[:1,:2])
describe(x[0,1])

indices = torch.LongTensor([0,2])
describe(torch.index_select(x,dim=1,index=indices)) ## Gets the 0 and 2 index columns of each row

indices = torch.LongTensor([0,0])
describe(torch.index_select(x,dim=0,index=indices))

row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0,1])
describe(x[row_indices,col_indices]) ## This gets [0,0] and [1,1]
describe(x[:2,:2]) ## This gets 4 square elements of the tensor

### Concatenating Tensors
describe(torch.cat([x,x],dim=0)) ## Appending them as rows

describe(torch.cat([x,x],dim=1)) ## Appending them as columns

describe(torch.stack([x,x])) ## Appends tensors along new dimension

### Linear Algebra operations on tensors

x1 = x.float()
x2 = torch.ones(3,2)
x2[:,1]+=1
describe(x2)
describe(torch.mm(x1,x2)) ## Matrix Multiplication

### Creating Tensors for gradient bookkeeping

x = torch.ones(2,2,requires_grad=True)
describe(x)
print(x.grad is None)

y = (x+2) * (x+5) + 3
describe(y)
print(y.grad is None)

z = y.mean()
describe(z)
z.backward()
print(x.grad is None)

### Using GPU for tensor operations. CUDA is provided by NVIDIA and 
### shouldn't look different than using a CPU tensor.
### This is example 1-16 in book, I could not follow because this 
### MacBook does not have an NVIDIA GPU

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.rand(3,3).to(device)
describe(x)

### Excercise 1-17 shows how to move tensors from and between GPU and CPU
### To work between machines.

### End of Chapter Excercises:

### Number 1: Create a 2D tensor and then add a dimension of size 1 inserted at dimension 0
a = torch.rand(3,3)
describe(a)
a.unsqueeze(0)
describe(a)
a.squeeze(0)
describe(a)

a2 = 3 + torch.rand(5,3) * (7-3)
describe(a2)
normalDist = torch.rand(3,3)
normalDist.normal_()
describe(normalDist)


