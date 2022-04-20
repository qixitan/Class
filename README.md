# *Classification*
####### some basic model of Classification
envs：python==3.8 torch>=1.8.1 GPU:NVIDIA RTX1080Ti
# *Note:* If you have any questions, please send an email to qixitan@qq.com or tanqixi508@gmail.com
#### *Project Process*
##### 2022/2/9 
Uploaded the classification file of resnet18 to CIFAR10 in exps.train_cifar10
##### 2022/2/26 
Upload function of separate training set and validation set  -- use the duplicated dataset class to implement training set, validation set separation and dataset creation after 2022/4/20
##### 2022/3/3  
Supplement Vgg. GoogLeNet, DenseNet and ConvNet,and We will continue to update the classification model.
##### 2022/4/18 
We will make a classification-based project showing the basic content of a project, including building dataset, Build the model, training process, and related log files
##### 2022/4/19 
The overall framework of the project is completed, and the data and model-related details can be defined by defining the overall content of the model and data in model.exp. And create an exp file in exps to achieve fine adjustment such as parameter adjustment. Training the model can be done through tools.train, just change the exp in tools.train to the exp file name in exps.default, for example, exps.default.ResNet18_cifar10 just need to set exp to ResNet18_cifar10;
The batch_size setting can be determined according to your GPU memory。
##### 2022/4/20
Added a dataset with the predicted flower category as an example: Implemented the dataset class in model.data.dataset.flowers_set, and rewrote the Exp function taking ResNet18 as an example to include in model.exp.flower_base and exps.default.ResNet18_flowers.
Training can be directly modified by modifying the exp in tools.train to ResNet18_flowers, and batch_size can be adjusted according to the size of your GPU memory. I use 1080Ti and can set it to 128
You can download data from: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
After downloading the dataset, extract it to the data file. The file directory is as follows:
>data
>>flowers

>>>daisy

>>>daandelion

>>>rose

>>>sunflower

>>>tulip
