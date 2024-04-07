# 数据安全监管课题组自研框架Bonfire-Core
中国科学院信息工程研究所-数据安全监管课题组，自研联邦学习算法框架Bonfire-Core，通过在单机上模拟服务器和客户端来验证算法的有效性，支持评价Non-IID算法和后门攻防算法。

## 数据处理
框架支持**标签不平衡**和**特征不平衡**两种数据情况。

其中标签不平衡通过将普通的平衡数据集使用Dirichlet分布重新分发来模拟，数据集详见`/dataset/label_data.py`，包括MNIST\CIFAR-10\CIFAR-100，分发部分见`/dataset/util.py`，不建议改动。

Non-IID程度可通过超参数$\alpha$调整，若选择任务3，则将`--alpha`调整为`0.05`以模拟Non-IID数据。若选择任务4，则调整为`100`，将数据恢复成IID的情况。

特征不平衡详见`/dataset/feature_data.py`，包括Digit-5、Office-Caltech-10和DomainNet，数据下载部分详见[FedBN](https://github.com/med-air/FedBN)。注意使用特征不平衡数据时，由于数据源数量固定，会覆盖参数中关于客户端数量的设定，如Digit-5有5个数据源。

## 可用模型
框架已集成简单的卷积神经网络用于实验的验证，位于`/nets`中，包含用于Digit-5数据集的DigitModel，用于CIFAR-10的FedAvgCNN等。

## 算法
所有联邦聚合和后门攻防算法均在`/method`中，需要继承`base_class.py`中的`Server`和`Client`基类，增加算法需要在`__init__.py`中注册后使用，建议以FedAvg和FedProx为模板实现自己的算法，需要覆盖的方法一般包括用于增加参数的`add_arguements`、用于训练的`train`以及聚合部分`aggregate`。

若选择任务4，则将方法`--method`改变为backdoor，参考`/method/backdoor.py`

## Tips
有问题请联系 wangshaopu@iie.ac.cn 和 xuanyuexin@iie.ac.cn，提问前请先自行阅读源代码，实在无法解决时请先阅读[提问的智慧](https://github.com/ryanhanwu/How-To-Ask-Questions-The-Smart-Way/blob/main/README-zh_CN.md)。