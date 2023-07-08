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
    
# show loss change    P
    
# train model    
def train(num_epochs):
    
    
if __name__ := "__main__":

```



## Part 2

#### a.方法部分

网络结构设计动机

​	卷积层：对图像进行特征提取，多层卷积能提取到更高级的特征，例如纹理、颜色、形状等，得到特征图。

​	池化层：对特征图进行采样，减小特征图的尺寸，保留重要的特征。

​	全连接层：将池化层输出的特征图，reshape为1维向量，进一步转化为更具有表达能力的特征表示，最后一层全连接层，将输入映射到10个分类的类别。



损失函数功能

​	例如，CrossEntropyLoss()，通过计算模型预测的概率分布与真实标签之间的交叉熵来度量预测结果的准确性，它输出一个概率分布，表示每个类别的概率，将这个概率分布与真实标签进行比较，并计算出交叉熵损失，衡量模型预测的不确定性与真实标签之间的差距。交叉熵损失越小，表示模型的预测结果越接近真实标签，模型的性能越好。

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
| loss_function        | CrossEntropyLoss |       |       |       |      |
| batch_size           | 64               | 128   | 256   | 32    |      |
| epoches              | 20               |       |       |       |      |
| acc                  | 68.88            | 67.83 | 67.34 | 68.39 |      |

| different learning rate |                  |       |       |        |      |
| ----------------------- | ---------------- | ----- | ----- | ------ | ---- |
| learning rate           | 0.001            | 0.002 | 0.01  | 0.0005 |      |
| optimizer               | Adam             |       |       |        |      |
| loss_function           | CrossEntropyLoss |       |       |        |      |
| batch_size              | 64               |       |       |        |      |
| epoches                 | 20               |       |       |        |      |
| acc                     | 68.88            | 68.89 | 57.12 | 67.89  |      |

| different acc |                  |       |       |         |      |
| ------------- | ---------------- | ----- | ----- | ------- | ---- |
| learning rate | 0.001            |       |       |         |      |
| optimizer     | Adam             | SGD   | AdamW | RMSprop |      |
| loss_function | CrossEntropyLoss |       |       |         |      |
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



进一步改进网络结构

```
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)
```

acc达到80.16%



#### c.结论部分

实验结论：不同learning rate 、loss func 、optimizer、net structure 下的loss收敛情况和acc 不同，

1.learning rate 过大则不收敛，过小则收敛很慢

2.实验发现，Adam + EntropyLoss 这一组合比较适合Cifar10数据集上的分类任务

3.在本实验中，对acc 和loss 收敛情况影响最大的，是网络结构:

​	2层全连接（acc=44%）

​	5层全连接（acc=49%）

​	3层卷积+1层池化+2层全连接（acc=75%）

​	（层卷+归一化）*4 + 池化 + 全连接 *2 (acc=80.16%)

## Part 3

使用gradio 

可视化loss, acc 的变化曲线

输入一个index，可视化图像，真实的label， predicted_label



（直接上图，如图有两个选项卡，一个show loss and acc, 另一个输入index, 显示图片信息 和 预测）

![image-20230706092736446](https://github.com/LebinDing749/code_practice/blob/cifar10/images/image-20230706092736446.png)

![image-20230706092937360](https://github.com/LebinDing749/code_practice/blob/cifar10/images/image-20230706092937360.png)

<img src="https://github.com/LebinDing749/code_practice/blob/cifar10/images/image-20230706092954081.png" alt="image-20230706092954081" style="zoom: 67%;" />
