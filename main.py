import torch.nn as nn

# First of all,we still need to consider the input sample data,
# and use it as the base for building the neural network

# define the input number
N=0 # numbers of groups witch is the test data

model = nn.Sequential(
    nn.Linear(N,10,True),
    nn.ReLU(),
    nn.Linear(10,5,True),
    nn.ReLU(),
    nn.Linear(5,3,True),
    nn.ReLU(),
    nn.Linear(3,2)
)

