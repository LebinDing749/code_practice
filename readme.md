# Practice 1 CIFAR-10 Dataset

## Part 1

自己实现网络结构，数据IO和训练部分的代码，未调用torchvision中的网络结构和数据处理等函数

代码结构：

```python

# data io
class Cifar10Dataset(Dataset):

# use plt show an image 
def show_image():

# my network    
class BinNet(nn.Module):
    
# show loss change    
def show_loss(train_losses):
    
# train model    
def train(num_epochs):
    
    
if __name__ := "__main__":

```



## Part 2

#### a.方法部分

网络结构设计动机

损失函数功能

网络结构图

#### b.实验部分

初步尝试

batch_size=64，epoch=20, 两层全连接 （acc 44%）,5层全连接（acc 48%），果断换网络结构

 

更换网络结构: （卷积 激活函数 池化）+ （卷积 激活函数 池化）+ 全连接

epoch 20

| different batch_size |                  |       |       |       |      |
| -------------------- | ---------------- | ----- | ----- | ----- | ---- |
| learning rate        | 0.001            |       |       |       |      |
| optimizer            | Adam             |       |       |       |      |
| loss_func            | CrossEntropyLoss |       |       |       |      |
| batch_size           | 64               | 128   | 256   | 32    |      |
| epoches              | 20               |       |       |       |      |
| acc                  | 68.88            | 67.83 | 67.34 | 68.39 |      |

| different learning rate |                  |       |       |        |      |
| ----------------------- | ---------------- | ----- | ----- | ------ | ---- |
| learning rate           | 0.001            | 0.002 | 0.01  | 0.0005 |      |
| optimizer               | Adam             |       |       |        |      |
| loss_func               | CrossEntropyLoss |       |       |        |      |
| batch_size              | 64               |       |       |        |      |
| epoches                 | 20               |       |       |        |      |
| acc                     | 68.88            | 68.89 | 57.12 | 67.89  |      |

| different acc |                  |       |       |         |      |
| ------------- | ---------------- | ----- | ----- | ------- | ---- |
| learning rate | 0.001            |       |       |         |      |
| optimizer     | Adam             | SGD   | AdamW | RMSprop |      |
| loss_func     | CrossEntropyLoss |       |       |         |      |
| batch_size    | 64               |       |       |         |      |
| epoches       | 20               |       |       |         |      |
| acc           | 68.88            | 62.87 | 68.71 | 68.66   |      |





算法改进

```
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
```

acc达到75.03%



images = images.float.to(device)

images = images.float().to(device)     增加精度有2%的提升

#### c.结论部分



## Part 3



