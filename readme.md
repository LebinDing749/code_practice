# Resnet-18

### 移除残差连接

自定义basicblock，移除残差连接

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out
```



训练结果： 可见还有和很大的提升空间

```shell
Epoch [1/10], Test Accuracy: 53.54%, Average_Loss: 1.5492
Epoch [2/10], Test Accuracy: 62.82%, Average_Loss: 1.3669
Epoch [3/10], Test Accuracy: 66.61%, Average_Loss: 1.2511
Epoch [4/10], Test Accuracy: 67.94%, Average_Loss: 1.1662
Epoch [5/10], Test Accuracy: 71.52%, Average_Loss: 1.0992
Epoch [6/10], Test Accuracy: 72.58%, Average_Loss: 1.0427
Epoch [7/10], Test Accuracy: 75.50%, Average_Loss: 0.9944
Epoch [8/10], Test Accuracy: 76.04%, Average_Loss: 0.9517
Epoch [9/10], Test Accuracy: 77.08%, Average_Loss: 0.9125
Epoch [10/10], Test Accuracy: 78.41%, Average_Loss: 0.8777
```

loss 和 acc 曲线

![image-20230708124050248](https://github.com/LebinDing749/code_practice/blob/resnet-18/images/img.png)

### 保留残差连接

定义残差块

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out
```

训练结果：

```shell
Epoch [1/10], Test Accuracy: 62.06%, Average_Loss: 1.2433
Epoch [2/10], Test Accuracy: 70.32%, Average_Loss: 1.0313
Epoch [3/10], Test Accuracy: 73.07%, Average_Loss: 0.8955
Epoch [4/10], Test Accuracy: 78.56%, Average_Loss: 0.7944
Epoch [5/10], Test Accuracy: 77.80%, Average_Loss: 0.7114
Epoch [6/10], Test Accuracy: 81.13%, Average_Loss: 0.6405
Epoch [7/10], Test Accuracy: 78.65%, Average_Loss: 0.5780
Epoch [8/10], Test Accuracy: 76.70%, Average_Loss: 0.5242
Epoch [9/10], Test Accuracy: 80.46%, Average_Loss: 0.4790
Epoch [10/10], Test Accuracy: 80.78%, Average_Loss: 0.4399
```

loss 和 acc 曲线:

![image-20230708132340274](https://github.com/LebinDing749/code_practice/blob/resnet-18/images/img_1.png)



观察不同：

1.本实验中，带残差连接的acc有略微提升。

2.带残差连接的，loss波动更小，收敛更稳定。