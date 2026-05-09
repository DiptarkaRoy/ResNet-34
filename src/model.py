import torch
import torch.nn as nn

# Residual Block for ResNet-34
class ResidualBlock(nn.Module):
    def __init__(self,in_channels, out_channels,stride=1, downsample=None):
        super(ResidualBlock,self).__init__()
        # First 3x3 convolutional layer
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        
        # Second 3x3 convolutional layer
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        # self.relu=nn.ReLU(inplace=True)

        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity = self.downsample(x)
        # Forward pass through the first convolutional layer
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        # Forward pass through the second convolutional layer
        out=self.conv2(out)
        out=self.bn2(out)

        # Skip connection
        out+=identity
        out=self.relu(out)

        # return output
        return out

# ResNet-34 Architecture
class ResNet34(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet34,self).__init__()
        self.in_channels=64

        # Initial Convolutional Layer
        self.conv1=nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        
        # Residual Layers
        self.layer1=self._make_layer(block, 64, layers[0], stride=1)
        self.layer2=self._make_layer(block, 128, layers[1], stride=2)
        self.layer3=self._make_layer(block, 256, layers[2], stride=2)
        self.layer4=self._make_layer(block, 512, layers[3], stride=2)

        # Final Output Layer
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512, num_classes)

        # --- INITIALIZATION ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # This is the "Zero-init" trick for the last BN in each residual branch
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample=None
        if stride!=1 or self.in_channels!=out_channels:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers=[]
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels=out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):

        # Initial Convolutional Layer
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out) 

        # Residual Layers
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)

        # Final Output Layer
        out=self.avgpool(out)
        # out=out.view(out.size(0), -1)
        out = torch.flatten(out, 1) # Cleaner than .view()
        out=self.fc(out)
        return out
    
# Function to create ResNet-34 model    
def resnet34(num_classes=1000)  ->  ResNet34:
    return ResNet34(ResidualBlock, [3, 4, 6, 3], num_classes)