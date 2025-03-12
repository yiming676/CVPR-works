import torch

import numpy as np

from part1 import device, dtype, loader_train, print_every, loader_val

'''
------------------------------------------------------------------------------------------------------------------------
Part II. Barebones PyTorch
PyTorch ships with high-level APIs to help us define model architectures conveniently, 
which we will cover in Part II of this tutorial. In this section, we will start with 
the barebone PyTorch elements to understand the autograd engine better. After this exercise, 
you will come to appreciate the high-level model API more.

We will start with a simple fully-connected ReLU network with two hidden layers and no biases 
for CIFAR classification. This implementation computes the forward pass using operations 
on PyTorch Tensors, and uses PyTorch autograd to compute gradients. It is important that you 
understand every line, because you will write a harder version after the example.

When we create a PyTorch Tensor with requires_grad=True, then operations involving that Tensor 
will not just compute values; they will also build up a computational graph in the background, 
allowing us to easily backpropagate through the graph to compute gradients of some Tensors with 
respect to a downstream loss. Concretely if x is a Tensor with x.requires_grad == True then 
after backpropagation x.grad will be another Tensor holding the gradient of x with respect to 
the scalar loss at the end.


PyTorch Tensors: Flatten Function
A PyTorch Tensor is conceptionally similar to a numpy array: it is an n-dimensional grid of numbers, 
and like numpy PyTorch provides many functions to efficiently operate on Tensors. As a simple 
example, we provide a flatten function below which reshapes image data for use in a fully-connected neural network.

Recall that image data is typically stored in a Tensor of shape N x C x H x W, where:

N is the number of datapoints
C is the number of channels
H is the height of the intermediate feature map in pixels
W is the height of the intermediate feature map in pixels
This is the right way to represent the data when we are doing something like a 2D convolution, 
that needs spatial understanding of where the intermediate features are relative to each other. 
When we use fully connected affine layers to process the image, however, 
we want each datapoint to be represented by a single vector -- it's no longer 
useful to segregate the different channels, rows, and columns of the data. 
So, we use a "flatten" operation to collapse the C x H x W values per representation into a 
single long vector. The flatten function below first reads in the N, C, H, and W values from a 
given batch of data, and then returns a "view" of that data. "View" is analogous to numpy's "reshape" 
method: it reshapes x's dimensions to be N x ??, where ?? is allowed to be anything (in this case, 
it will be C x H x W, but we don't need to specify that explicitly).
'''

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flattening: ', x)
    print('After flattening: ', flatten(x))

test_flatten()

'''
Barebones PyTorch: Two-Layer Network
Here we define a function two_layer_fc which performs the forward pass of a two-layer fully-connected ReLU
 network on a batch of image data. After defining the forward pass we check that it doesn't crash and that 
 it produces outputs of the right shape by running zeros through the network.

You don't have to write any code here, but it's important that you read and understand the implementation.
'''

import torch.nn.functional as F  # useful stateless functions


def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass;
    PyTorch will take care of the backward pass for us.

    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).

    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]

    w1, w2 = params

    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    # you can also use `.clamp(min=0)`, equivalent to F.relu()
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x


def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 50), dtype=dtype)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    scores = two_layer_fc(x, [w1, w2])
    print(scores.size())  # you should see [64, 10]


two_layer_fc_test()

'''
Barebones PyTorch: Three-Layer ConvNet
Here you will complete the implementation of the function three_layer_convnet, which will perform the 
forward pass of a three-layer convolutional network. Like above, we can immediately test our 
implementation by passing zeros through the network. The network should have the following architecture:

1. A convolutional layer (with bias) with channel_1 filters, each with shape KW1 x KH1, and zero-padding of two
2. ReLU nonlinearity
3. A convolutional layer (with bias) with channel_2 filters, each with shape KW2 x KH2, and zero-padding of one
4. ReLU nonlinearity
5. Fully-connected layer with bias, producing scores for C classes.

HINT: For convolutions: http://pytorch.org/docs/stable/nn.html#torch.nn.functional.conv2d; pay attention 
to the shapes of convolutional filters!
'''


def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None
    ################################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.                #
    ################################################################################
    # First convolutional layer with ReLU nonlinearity
    x = F.conv2d(x, conv_w1, bias=conv_b1, padding=2)  # shape: [N, channel_1, H, W]
    x = F.relu(x)

    # Second convolutional layer with ReLU nonlinearity
    x = F.conv2d(x, conv_w2, bias=conv_b2, padding=1)  # shape: [N, channel_2, H-2, W-2]
    x = F.relu(x)

    # Flatten the output for the fully connected layer
    N = x.shape[0]
    x = x.view(N, -1)  # shape: [N, channel_2 * (H-2) * (W-2)]

    # Fully connected layer
    scores = torch.matmul(x, fc_w) + fc_b  # shape: [N, C]

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################
    return scores

'''
After defining the forward pass of the ConvNet above, run the following cell to test your implementation.

When you run this function, scores should have shape (64, 10).
'''

def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]
three_layer_convnet_test()

'''
Barebones PyTorch: Initialization
Let's write a couple utility methods to initialize the weight matrices for our models.

random_weight(shape) initializes a weight tensor with the Kaiming normalization method.
zero_weight(shape) initializes a weight tensor with all zeros. Useful for instantiating bias parameters.
The random_weight function uses the Kaiming normal initialization method, described in:

He et al, Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, 
ICCV 2015, https://arxiv.org/abs/1502.01852
'''

def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

# create a weight of shape [3 x 5]
# you should see the type `torch.cuda.FloatTensor` if you use GPU.
# Otherwise it should be `torch.FloatTensor`
random_weight((3, 5))

'''
Barebones PyTorch: Check Accuracy
When training the model we will use the following function to check the accuracy of our model 
on the training or validation sets.

When checking accuracy we don't need to compute any gradients; as a result we don't need PyTorch 
to build a computational graph for us when we compute scores. To prevent a graph from being built 
we scope our computation under a torch.no_grad() context manager.
'''


def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.

    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model

    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

'''
BareBones PyTorch: Training Loop
We can now set up a basic training loop to train our network. We will train the model using 
stochastic gradient descent without momentum. We will use torch.functional.cross_entropy to 
compute the loss; you can read about it here.

The training loop takes as input the neural network function, a list of initialized parameters 
([w1, w2] in our example), and learning rate.
'''


def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.

    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD

    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()


'''
BareBones PyTorch: Train a Two-Layer Network
Now we are ready to run the training loop. We need to explicitly allocate tensors for the fully 
connected weights, w1 and w2.

Each minibatch of CIFAR has 64 examples, so the tensor shape is [64, 3, 32, 32].

After flattening, x shape should be [64, 3 * 32 * 32]. This will be the size of the first dimension of w1. 
The second dimension of w1 is the hidden layer size, which will also be the first dimension of w2.

Finally, the output of the network is a 10-dimensional vector that represents the probability 
distribution over 10 classes.

You don't need to tune any hyperparameters but you should see accuracies above 40% after 
training for one epoch.
'''

hidden_layer_size = 4000
learning_rate = 1e-2

w1 = random_weight((3 * 32 * 32, hidden_layer_size))
w2 = random_weight((hidden_layer_size, 10))

train_part2(two_layer_fc, [w1, w2], learning_rate)


'''
BareBones PyTorch: Training a ConvNet
In the below you should use the functions defined above to train a three-layer convolutional network on 
CIFAR. The network should have the following architecture:

1. Convolutional layer (with bias) with 32 5x5 filters, with zero-padding of 2
2. ReLU
3. Convolutional layer (with bias) with 16 3x3 filters, with zero-padding of 1
4. ReLU
5. Fully-connected layer (with bias) to compute scores for 10 classes
You should initialize your weight matrices using the random_weight function defined above, and you should
 initialize your bias vectors using the zero_weight function above.

You don't need to tune any hyperparameters, but if everything works correctly you should achieve
 an accuracy above 42% after one epoch.
'''

learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

################################################################################
# TODO: Initialize the parameters of a three-layer ConvNet.                    #
################################################################################
conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight(channel_1)
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight(channel_2)

# Calculate the size of the flattened tensor after the second convolutional layer
fc_input_size = channel_2 * 10 * 10  # CIFAR-10 images are 32x32, after two convolutions they are 30x30
fc_w = random_weight((fc_input_size, 10))
fc_b = zero_weight(10)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
train_part2(three_layer_convnet, params, learning_rate)









