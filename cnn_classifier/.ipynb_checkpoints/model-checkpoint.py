from imports import *
#Model-class
class NeuralNetwork(nn.Module):
    """
    We use standard CNN for this classification, without any tricks from ResNet, MobileNet or Inception. Maybe (?) these model will be
    rewritted with only one convolution block in different parameters. Bias in this model increase converge. But it may be a little bit 
    overfitting. 
    """
    __channels_list__ = [3, 16, 32, 64, 128, 256, 512]  # List of in/out channels for convolutions. If you have a lot of memory on GPU you can expand it.
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        def __block__(in_channels, out_channels):
            """
            This architecture of Neural Networks is obvious, I guess. BTW, without BN model converge in slowly in 3-4 times.
            """
            return nn.Sequential(
             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=True),
             nn.ReLU(),
             nn.MaxPool2d(2, 2),
             nn.BatchNorm2d(out_channels)
             )

        self.__conv_list__ = [__block__(self.__channels_list__[i], self.__channels_list__[i+1]) for i in range(len(self.__channels_list__)-1)]

        self.conv_stack = nn.Sequential()
        
        for conv in self.__conv_list__:
            self.conv_stack.extend(conv)
    
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 120),
            nn.Linear(120, 84),
            nn.Dropout(dropout_rate),
            nn.Linear(84, classnum),
        )
        
    def forward(self, x):
        for conv in self.conv_stack:
            x = conv(x)
        x = self.linear_stack(x)
        return x
