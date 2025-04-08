import torch.nn as nn
import torch.nn.functional as F
import torch

# setting up the CNN architecture
# 2 conv layers
# first layer has 64 distinct filters of size 5x5
# second layer also has 64 filters with kernel size 5x5
class mnist_conv(nn.Module):
    def __init__(self, mean = 0.0, std = 1.0):
        super(mnist_conv, self).__init__()
        self.act_fn = nn.ReLU()
        #self.act_fn = nn.LeakyReLU(negative_slope=0.01)
        #self.act_fn = nn.Softplus()
        
        #
        self.conv = torch.nn.Conv2d
        self.linear = torch.nn.Linear
        self.mean = mean
        self.std = std

        self.layers1 = []
        self.layers2 = []
        self.layers1.append(self.conv(1, 64, 5)) # 1 input channel, 64 output channels (i.e no of kernels), 5x5 kernel
        self.layers1.append(nn.MaxPool2d(2))
        self.layers1.append(self.act_fn)

        self.layers1.append(self.conv(64, 64, 5))
        self.layers1.append(nn.MaxPool2d(2))
        self.layers1.append(self.act_fn)

        self.layers1.append(nn.Flatten())

        self.layers2.append(self.linear(4 * 4 * 64, 128)) # self.layers2 refers to the fully connected layers, we have 2 of them
        self.layers2.append(self.act_fn)

        self.layers2.append(self.linear(128, 10))
        #self.layers2.append(torch.nn.Softmax(dim=1))        

        self.layers1 = nn.Sequential(*self.layers1)
        self.layers2 = nn.Sequential(*self.layers2)

    def forward(self, x):
        x = (x - self.mean)/self.std
        x = self.layers1(x)
        return self.layers2(x)


"""
Here’s the flow from input to output, layer by layer:

Input Normalization

Subtract mean, divide by std.
First Convolution

Convolution: Conv2d(1, 64, kernel_size=5) generates 64 feature maps from the single input channel.
Max-Pooling: MaxPool2d(2) halves spatial dimensions.
Activation: ReLU.
Second Convolution

Convolution: Conv2d(64, 64, kernel_size=5) keeps 64 channels.
Max-Pooling: MaxPool2d(2) halves dimensions again.
Activation: ReLU.
Flatten

Transforms the pooled feature map (now 4×4×64=1024 elements) into a single vector.
Fully Connected Layers

Linear(1024 → 128), ReLU.
Linear(128 → 10).
Output

Returns a 10-dimensional output (logits), one per MNIST digit class.
"""