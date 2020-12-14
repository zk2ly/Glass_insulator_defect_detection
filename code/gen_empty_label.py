import os

imgs = os.listdir('/home/zk/darknet/insulator_data/train/JPEGImages/')

for img in imgs:
    label_path = '/home/zk/darknet/insulator_data/train/labels/' + img.split('.')[0] + '.txt'
    if not os.path.exists(label_path):
        f = open(label_path, 'w')
        f.close()