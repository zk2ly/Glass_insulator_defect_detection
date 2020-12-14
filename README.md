# 玻璃绝缘子检测
- [玻璃绝缘子检测](#玻璃绝缘子检测)
  - [数据准备](#数据准备)
  - [训练](#训练)
  - [测试](#测试)
  - [效果](#效果)
  - [To do](#to-do)
  - [经验总结](#经验总结)

## 数据准备

30张6000x6000以上的高分辨率图片

选取7张作为测试集    **测试集背景信息应该要多样**

![原图](https://github.com/zk2ly/Glass_insulator_defect_detection/blob/main/README_IMAGES/1.png)

使用YOLT的思想，将训练集/测试集的每一张图都裁剪成608x608的小图，命名格式：“原图名称  _ 左上角坐标 _ 图像大小 _ 原图大小"

如“003_0_0_608_608__6016_4016.JPG”  

YOLT论文：https://arxiv.org/abs/1805.09512

![裁剪后的小图](https://github.com/zk2ly/Glass_insulator_defect_detection/blob/main/README_IMAGES/2.png)

对训练集/测试集裁剪后的小图做标注，得到xml文件，转成yolo格式  

此时，

训练集图片：2692张    标注：156个

测试集图片：666张  标注：41个

对没有目标的负样本，使用空白的txt文件做标注

用albumentations对训练集含有标注的156个图像做数据增强:随机翻转，随机旋转90°，亮度对比度变化，缩小后填充黑边再随机旋转0-90°，最终得到图片5188张，负样本正样本比例大约1：1。

albumentations : https://github.com/zk2ly/How-to-use-Albumentations


## 训练

使用darknet版yolov4训练 :  https://github.com/AlexeyAB/darknet

两块2080训练  先用k-means聚类 网络输入416x416 batch=16  sub=8  再预训练模型上跑8000个epoch  准确率达到97%


## 测试

把一张图裁剪成608x608的若干小图依次放入网络进行检测，最终检测框的坐标加上图片所属左上角的坐标得到在原图中的绝对坐标，然后对原图上的所有框做一个nms，得到最终的检测图。


## 效果

![效果图样例](https://github.com/zk2ly/Glass_insulator_defect_detection/blob/main/README_IMAGES/3.png)


## 泛化
在“数字电网开发者大会——小样本输电线路玻璃绝缘子自爆程度识别”训练集上的泛化效果

![泛化效果图样例](https://github.com/zk2ly/Glass_insulator_defect_detection/blob/main/README_IMAGES/4.png)


## To do

- [ ] 图片串行输入改为并行，提升速度
- [ ] 检测时提高网络的分辨率，提升精度
- [ ] 检测时裁剪为比608x608更大的小图，减少小图数目，提高速度


## 经验总结

1.最开始考虑速度，用得yolov4-tiny，效果很不好，换成yolov4后效果变好，师姐说要**优先考虑精度，然后优化速度。**

2.数据增强时尝试过模糊，噪声等，效果不好，最后问师姐，原因是图片本身很高清，不用加模糊和噪声。

**图像增强应该符合目标本身的特征，即目标有大小，方向，亮度变化，那么增强时也可以做这些变化，目标并不模糊，增强时就不做模糊。**

3.最开始划分训练集和测试集的两种方案效果都不好：

(1)裁剪全部30张图，然后对197个正样本做增强使正负样本1:1，然后随机划分训练集和测试集；

(2)裁剪全部30张图，然后对197个正样本做增强使正负样本1:1，然后把原始正样本做测试集，增强得到的正样本加原始负样本做训练集；

这两种情况都和真实的图片情况不一致，真实的图片裁剪后，正样本只是极少的部分，5%左右，而第一种随机划分使测试集的正样本达到50%，第二种更是只有正样本，因此网络训练时就偏离了实际情况。

**最开始划分数据集，验证集，测试集时，要在原图上做划分，训练方法如数据增强等，只能用在训练集上。**

4.按照darknet主页的说法，没有目标的负样本和有目标的正样本，训练时最好1:1，并且为负样本添加空的txt文件作为标注。













