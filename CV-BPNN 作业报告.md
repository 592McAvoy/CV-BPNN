# CV-BPNN 作业报告


罗宇辰 516030910101

---

## 作业概况 ##

 - 实现语言：python 
 - 工具库：numpy，pandas 
 - 实验数据：数据集[iris][1] 
 - 报告：CV-BPNN作业报告
 - 代码文件：
 1. test.py：程序入口文件
 2. BPNN.py：隐藏层、BPNN定义
 3. util.py：激活函数、批量归一化函数定义


----------


## 简介 ##
本次作业中实现了一个简易BPNN网络：

 1. 隐藏层中对传入的数据依次进行线性分类、批量归一化、激活，然后输出数据 
 2. 每一次训练过程中计算折叶损失
 3. 通过反向传播计算参数梯度，采用梯度下降的方法更新参数
 
其中，网络的隐藏层数，隐藏层节点数，学习率等参数均可变

 


----------


## 实现介绍 ##
### 1. test.py ###

    def loadfile():
        ...//load iris.txt
        
    if __name__ == "__main__":
        data, label = loadfile()
    
        net = BPNN.Net()
        ...//设置net结构
    
        #  train the net
        iter_num = 150
        for i in range(iter_num):
            loss, score = net.train(delta=1, reg=0.3, learning_rate=0.05)
        ...
 在main函数中可以手动对*learning_rate*，*layer_num*，*layer_points*，*delta*等参数进行设置，整个程序将对net进行*iter_num*次更新          

### 2. BPNN.py ###

（1）隐藏层类

    class Layer:
    """
    Hidden Layer
        forward:        input -> [linear -> batch-normalize -> activate] -> output
        backward:       dout -> [actBack -> BNBack -> lineBack] -> dx
        update:         gradient descent
    """

初始化过程中：对权重W采用了随机数初始化，并除以输入数据量校验偏差；对偏置b采用了零初始化
    
        def __init__(self, in_size, out_size):
            self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size)
            self.b = np.zeros(out_size)
            ...//other initialization
前向传播经过线性分类、批量归一化和激活三个阶段。一开始是没有归一化处理的，但是在sigmoid函数中会出现溢出，使用ReLU更是会出现中间数据过大溢出的情况，所以加入了归一化处理
    
        def forward(self, x):
            #  linear
            ...
    
            #  BN
            ...
    
            #  Activate
                # 1.sigmoid
                #  out = self.activate.sigmoidFoward(norm)
        
                # 2.tanh
                #  out = self.activate.tanhFoward(norm)
        
                # 3.ReLU
                out = self.activate.ReLUFoward(norm)
            ...
            return out
反向传播过程中计算各参数的梯度
    
        def backward(self, dout):
            ...//计算BN，激活函数等输出梯度
            self.dw = np.dot(x.T, dtmp)
            self.db = np.sum(dtmp, axis=0)
            dx = np.dot(dtmp, self.W.T)
            return dx
使用梯度下降方法进行参数更新

        def update(self, lr):
            self.W = self.W - self.dw * lr
            self.b = self.b - self.db * lr
    
（2）BPNN网络

    class Net:
    """
    BPNN Net
        setData:        set train data
        setLayer:       set amount of hidden layers and hidden points
        train:          train the net
    """
    
设置训练数据，由于iris数据集形式为前N-1列为数据，最后一列为分类标签，所以将输入数据存储为训练数据data，和计算损失的目标依据target

    def setData(self, input, labels):
        ...
        for i in range(N):
            label = input[i, D-1]
            target[i] = labels.index(label)
        self.labels = labels
        self.target = target
根据传入参数初始化隐藏层，并初始化输出层权重计算矩阵W
    
    def setLayers(self, layer_num, layer_points=2):
        ...
        hidden_layers = []
        hidden_layers.append(Layer(D, layer_points))
        for i in range(layer_num-1):
            hidden_layers.append(Layer(layer_points, layer_points))
        self.hidden_layers = hidden_layers
        self.W = np.random.randn(layer_points, len(self.labels))

训练BPNN网络：训练数据经过多个隐藏层后，在输出层计算得到形状为（N,class）的score矩阵，即每个输入数据关于不同标签的得分，根据这个分类得分和数据真实的类别计算折叶损失，然后反向传播更新网络参数

    def train(self, delta=1, reg=0.3, learning_rate=0.03):
        ...

        //训练数据依次经过隐藏层
        prev = hiden[0].forward(self.data)
        for i in range(1, layerNum):
            prev = hiden[i].forward(prev)
            
        //得到score矩阵
        #  score shape - (N, class)
        score = np.dot(prev, self.W)

        """
        Hinge Loss  ->   loss = 1/2 * w^2 + SUM(max(s_i - correct + delta, 0))
        """
        ...//计算折叶损失loss，和梯度dw，dprev

        //更新网络参数
        #  update W
        self.W -= dW * learning_rate

        #  update hidden layers
        for i in range(layerNum, 0, -1):
            dprev = hiden[i-1].backward(dprev)
            hiden[i - 1].update(learning_rate)
        self.hidden_layers = hiden

        return loss, score

### 3. util.py ###

(1)批量归一化处理

    class BN:
    """
    Batch Normalization     out = (x - mean(x)) / variance(x)
    """
前向传播，反向传播方法

    def forward(self, x, eps=1e-4):
        ...
        return out

    def backward(self, dout):
        ...
        return dx

(2)激活函数


    class Activate:
    """
    Activation Function
        1.Sigmiod   out = 1 / (1 - exp(-x))
        2.Tanh      out = (exp(x)-exp(-x) / (exp(x)+exp(-x)
        3.ReLU      out = max(0, x)
    """

除了课程要求的logistic（sigmoid）函数之外，测试了tanh和ReLU函数

   
    def sigmoidFoward(self, x):

    def sigmoidBackward(self, dout):
        
    def tanhFoward(self, x):
       
    def tanhBackward(self, dout):
        
    def ReLUFoward(self, x):
       
    def ReLUBackward(self, dout):


----------
## 结果展示 ##
设置参数：

    layer_num=100, layer_points=8 //100个隐藏层，每层8个节点
    delta=1, reg=0.3, learning_rate=0.05
    iter_num = 150  //迭代150次
                    //激活函数为ReLU
输出结果：

    iter 0:	     loss:  [6.32312132]
    iter 10:	 loss:  [4.89704039]
    iter 20:	 loss:  [4.06694371]
    iter 30:	 loss:  [3.43658893]
    iter 40:	 loss:  [2.99801756]
    iter 50:	 loss:  [2.69888273]
    iter 60:	 loss:  [2.48037379]
    iter 70:	 loss:  [2.30860113]
    iter 80:	 loss:  [2.1845839]
    iter 90:	 loss:  [2.09322956]
    iter 100:	 loss:  [2.03293347]
    iter 110:	 loss:  [1.97298013]
    iter 120:	 loss:  [1.9379504]
    iter 130:	 loss:  [1.91202203]
    iter 140:	 loss:  [1.89024119]
图像：![loss figure][2]


  [1]: https://archive.ics.uci.edu/ml/datasets/Iris
  [2]: http://m.qpic.cn/psb?/V13Ti98m05LW5b/zAUHWsOU4p4hIAx.XFtyKGSAA009bzPJfMRRHPZaIJo!/b/dFIBAAAAAAAA&bo=gALgAQAAAAADB0E!&rf=viewer_4