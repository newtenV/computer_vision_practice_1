# 神经网络可视化工具netron

### netron介绍

 Netron是一种用于神经网络、深度学习和机器学习模型的可视化工具，它可以为模型的架构生成具有描述性的可视化(descriptive visualization)。源码在：https://github.com/lutzroeder/netron ，主要由JavaScript语言实现。

Netron是一个跨平台工具，可以在Linux、Windows和Mac上运行，并且支持多种框架和格式。Netron支持ONNX、TensorFlow Lite、Caffe、Keras、Darknet、PaddlePaddle、ncnn、MNN、Core ML、RKNN、MXNet、MindSpore Lite、TNN、Barracuda、Tengine、CNTK、TensorFlow.js、Caffe2 和 UFF。它还实验性支持PyTorch、TensorFlow、TorchScript、OpenVINO、Torch、Vitis AI、kmodel、Arm NN、BigDL、Chainer、Deeplearning4j、MediaPipe、ML.NET 和 scikit-learn。

### netron使用方法

(1)直接下载：第一种是以软件的方式安装netron，然后打开软件载入模型，下载地址见[github主页](https://link.zhihu.com/?target=https://github.com/lutzroeder/netron)。

(2)作为python库进行安装，在python代码调用netron库来载入模型进行可视化。 可以通过 pip install netron进行安装。

(3)直接使用netron在线网站，直接上传模型文件查看可视化结果：

[netron在线网站]: https://netron.app/

效果如下：

![netron](C:\my\works\计算机视觉实践-练习\计算机视觉实践-练习2\images\netron.png)



# 手写数字识别

### MNIST数据集介绍

MNIST 数据库是一个大型手写数字数据库（包含0~9十个数字），包含 60,000 张训练图像和 10,000 张测试图像，通常用于训练各种图像处理系统。训练数据集取自美国人口普查局员工，而测试数据集取自美国高中生。所有的手写数字图片的分辨率为28*28。

### MNIST数据集下载

下载文件，解压它们，并将它们存储在工作区目录`./MNIST_Data/train`和`./MNIST_Data/test`.

```python
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
            train-images.idx3-ubyte
            train-labels.idx1-ubyte

```



### 定义网络

LeNet网络相对简单。除了输入层之外，LeNet网络还有七层，包括两个卷积层、两个下采样层（池化层）和三个全连接层。每层包含不同数量的训练参数，如下图所示：

![LeNet_5](C:\my\works\计算机视觉实践-练习\计算机视觉实践-练习2\images\LeNet_5.jpg)

根据LeNet网络结构，定义网络层如下：

```python
import mindspore.ops.operations as P

class LeNet5(nn.Cell):
    """
    Lenet network structure
    """
    #define the operator required
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

#use the preceding operators to construct networks
def construct(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.max_pool2d(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.max_pool2d(x)
    x = self.reshape(x, (self.batch_size, -1))
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x
```



### 定义损失函数和优化器

`SoftmaxCrossEntropyWithLogits`为本例中使用了损失函数

```
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

 net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
```



本例中使用了流行的 Momentum 优化器。

```
lr = 0.01
momentum = 0.9
#create the network
network = LeNet5()
#define the optimizer
net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
```



### 训练网络

##### 保存配置的模型

```
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# set parameters of check point
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10) 
# apply parameters of check point
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck) 
```

##### 配置网络训练

使用`model.train`MindSpore提供的API可以轻松训练网络。在此示例中，设置`epoch_size`为 1 以训练数据集五次迭代。

```
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model


def train_net(args, model, epoch_size, mnist_path, repeat_size, ckpoint_cb, sink_mode):
    """define the training method"""
    print("============== Starting Training ==============")
    #load training dataset
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode) # train



epoch_size = 1    
mnist_path = "./MNIST_Data"
repeat_size = epoch_size
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(args, model, epoch_size, mnist_path, repeat_size, ckpoint_cb, dataset_sink_mode)

```



训练结果文件示例

```
checkpoint_lenet-1_1875.ckpt
```



### 验证模型

得到模型文件后，我们验证模型的泛化能力。

```
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def test_net(args,network,model,mnist_path):
    """define the evaluation method"""
    print("============== Starting Testing ==============")
    #load the saved model for evaluation
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    #load parameter to the network
    load_param_into_net(network, param_dict)
    #load testing dataset
    ds_eval = create_dataset(os.path.join(mnist_path, "test")) # test
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))

if __name__ == "__main__":
    ...
    test_net(args, network, model, mnist_path)

```



结果输出

![results](C:\my\works\计算机视觉实践-练习\计算机视觉实践-练习2\images\result.png)