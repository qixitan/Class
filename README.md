# *Classification*
  some basic model of Classification
  envs：python==3.8 torch>=1.8.1 GPU:NVIDIA RTX1080Ti

#### *Project Process*
##### 2022/2/9 
  Uploaded the classification file of resnet18 to CIFAR10 in exps.train_cifar10
##### 2022/2/26 
  Upload function of separate training set and validation set  -- use the duplicated dataset class to implement training set,   validation set separation and dataset creation after 2022/4/20
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


├─data

│  ├─flowers

│  │  ├─daisy

│  │  ├─dandelion

│  │  ├─rose

│  │  ├─sunflower

│  │  └─tulip

  In addition to supplement the ViT。
  
##### 2020/4/21
Updated various versions of ViT, such as ViT_Ti, ViT_S, ViT_B, ViT_L. and gives some parameters of the model(*The values in parentheses are expressed as pytorch official data*) as follow.
<table>
	<tr>
	  <td>Model_Name</td>
	  <td>Params_Size (MB)</td>
	  <td>Image_Size</td>
	</tr>
	<tr>
	  <td>resnet18</td>
	  <td> 44.59 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>resnet34</td>
	  <td> 83.15 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>resnet50</td>
	  <td> 97.49 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>resnet101</td>
	  <td> 169.94 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>resnet152</td>
	  <td> 229.62 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg11</td>
	  <td> 506.83 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg11_bn</td>
	  <td> 506.85 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg13</td>
	  <td> 507.54 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg13_bn</td>
	  <td> 507.56 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg16</td>
	  <td> 527.79 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg16_bn</td>
	  <td> 527.82 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg19</td>
	  <td> 548.05 </td>
	  <td>(224, 224)</td>
	</tr>
	<tr>
	  <td>vgg19_bn</td>
	  <td> 548.09 </td>
	  <td>(224, 224)</td>
	 </tr>
	 <tr>
	   <td>GoogLeNet</td>
	   <td> 27.86(49.61) </td>
	   <td>(224, 224)</td>
	  </tr>
	  <tr>
	   <td>DenseNet121</td>
	   <td> 41.47 </td>
	   <td>(224, 224)</td>
	  </tr>
	  <tr>
	   <td>ViT_Ti</td>
	   <td> 21.67 </td>
	   <td>(224, 224)</td>
	  </tr>
	  <tr>
	   <td>ViT_S</td>
	   <td> 83.83 </td>
	   <td>(224, 224)</td>
	  </tr>
	  <tr>
	   <td>ViT_B</td>
	   <td> 329.65 </td>
	   <td>(224, 224)</td>
	  </tr>
	  <tr>
	   <td>ViT_L</td>
	   <td> 1160.14 </td>
	   <td>(224, 224)</td>
	  </tr>
	</tr>
</table>

**Note: If you have any questions, please send an email to qixitan@qq.com or tanqixi508@gmail.com**
