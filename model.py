import  torch.nn     as  nn
import  torch.nn.functional as  F

class   CNN(nn.Module):
    """Defines a simple CNN model for MNIST classification"""

    def __init__(self):
        super(CNN,  self).__init__()
        self.conv1  =   nn.Conv2d(1,    32, kernel_size=3)      #First convolutional layer
        self.conv2  =   nn.Conv2d(32,   64, kernel_size=3)      #Second convolutional layer
        self.fc1    =   nn.Linear(9216, 128)                    #Fully connected layer
        self.fc2    =   nn.Linear(128,  10)                     #Output layer with 10 classes
    
    def forward(self,   x):
        x   =   F.relu(self.conv1(x))                           #Apply ReLU activation
        x   =   F.relu(self.conv2(x))
        x   =   F.max_pool2d(x, 2)                              #Apply max pooling
        x   =   x.view(-1,  9216)                               #Flatten the feature maps
        x   =   F.relu(self.fc1(x))
        x   =   self.fc2(x)                                     #Output layer
        return  F.log_softmax(x,    dim=1)                      #Log softmax for classification