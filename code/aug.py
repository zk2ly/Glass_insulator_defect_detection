import albumentations as A
import os
from tqdm import tqdm
import cv2
import random 

transform = A.Compose(
    [   
        A.Flip(p=0.5),    
        A.RandomRotate90(p=0.5),   
        A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
         
        # A.RandomSizedBBoxSafeCrop (608, 608, erosion_rate=0.0, interpolation=1, always_apply=False, p=0.5), 
        # A.OneOf([
        #     A.GaussNoise (var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
        #     A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5) 
        # ], p=0.5),
        # A.OneOf([
        #     A.GaussianBlur (blur_limit=(1,3), sigma_limit=0, always_apply=False, p=0.5),
        #     A.MotionBlur(blur_limit=3, always_apply=False, p=0.5)
        # ], p=0.5)
        A.Compose([
            A.Resize(448, 448, interpolation=0, always_apply=False, p=1),
            A.PadIfNeeded(min_height=608, min_width=608, border_mode=0, p=1),
            A.Rotate(limit=90, interpolation=1, border_mode=0, value=None, mask_value=None, always_apply=False, p=1)
        ], p=0.5)
    ], 
    bbox_params=A.BboxParams(format='yolo')
)


aug_pic_num = 2500
labels = os.listdir('train/labels/')
epochs = int(round(aug_pic_num / len(labels)))

for epoch in tqdm(range(epochs)):
    for idx, label in  enumerate(labels):
        bboxes = []
        trans_bboxes = []
        label_path = 'train/labels/' + label
        img_path = 'train/JPEGImages/' + label.split('.')[0] + '.JPG'
        image = cv2.imread(img_path)
        with open(label_path, 'r') as f:
            for line in f:
                lst = line.split(' ')
                bboxes.append([float(lst[1]), float(lst[2]), float(lst[3]), float(lst[4]), lst[0]])
        n = random.randint(0, 1e99)
        random.seed(n)
        transformed = transform(image=image, bboxes=bboxes)
        image_outpath = 'train/aug_images/' + label.split('.')[0] + '_' + str(epoch) + '_' + str(idx) + '.jpg'
        label_outpath = 'train/aug_labels/' + label.split('.')[0] + '_' + str(epoch) + '_' + str(idx) + '.txt'
        cv2.imwrite(image_outpath, transformed['image'])
        with open(label_outpath, 'w') as f:
            for box in transformed['bboxes']:
                box_ = [str(i) for i in box]
                box_.insert(0, box_.pop())
                f.write(' '.join(box_) + '\n')
        
    
        
    
