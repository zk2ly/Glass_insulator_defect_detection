import os
import random

train_file = '/home/zk/darknet/insulator_data/train.txt'
val_file = '/home/zk/darknet/insulator_data/valid.txt'

train = open(train_file, 'a')
val = open(val_file, 'a')

train_imgs = os.listdir('/home/zk/darknet/insulator_data/train/JPEGImages/')
val_imgs = os.listdir('/home/zk/darknet/insulator_data/valid/JPEGImages/')

for img in train_imgs:
    img_path = '/home/zk/darknet/insulator_data/train/JPEGImages/' + img
    train.write(img_path + '\n')

        
for img in val_imgs:
    img_path = '/home/zk/darknet/insulator_data/valid/JPEGImages/' + img
    val.write(img_path + '\n')