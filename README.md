# Classification
some basic model of Classification

envs：
{
python==3.8
torch>=1.8.1
}


##### 2022/2/9 
上传了resnet18对CIFAR10的分类文件
##### 2022/2/26 
上传分离训练集和验证集函数
##### 2022/3/3  
重新整理 并增加了vgg。googlenet、densenet。
##### 2022/4/18 
持续更新 打算做成一个开源的以图像分类为例子 展示一个项目的基本内容  
##### 2022/4/19 
继续更新 现在可以通过tools.train 直接运行  其他使用其他模型可以在model.exp中根据自己需要复写的使用自己数据集和模型类就可以了
##### 2022/4/20
新增了以预测花的类别为例子的dataset 在model.data.dataset.flowers_set中实现了dataset类
并重写了以ResNet18为例子的Exp函数包括在model.exp.flower_base以及exps.default.ResNet18_flowers
训练可直接通过修改tools.train中exp修改为ResNet18_flowers、batch_size可根据自己GPU内存大小调整 本人使用1080Ti可设置为128

数据集下载地址为：https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
数据集下载后解压到data文件下即可 文件目录如下
>data
>>flowers

>>>daisy

>>>daandelion

>>>rose

>>>sunflower

>>>tulip
