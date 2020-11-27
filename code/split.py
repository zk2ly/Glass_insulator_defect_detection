import os
import cv2
import time
from tqdm import tqdm

def slice_im(image_path, outdir, sliceHeight=608, sliceWidth=608, overlap=0.15):
    image0 = cv2.imread(image_path)  
    name = image_path.split('/')[-1].split('.')[0]
    ext = '.' + image_path.split('/')[-1].split('.')[1]
    win_h, win_w = image0.shape[:2]

    dx = int((1. - overlap) * sliceWidth)  # overlap代表重叠区域 (1-overlap代表需要移动的距离)
    dy = int((1. - overlap) * sliceHeight)

    for y0 in range(0, image0.shape[0], dy):
        for x0 in range(0, image0.shape[1], dx):
            
            # 如果此时左上角坐标加上裁剪大小已经超出了图片，那么修改左上角的坐标
            if y0+sliceHeight > image0.shape[0]:  
                y0 = image0.shape[0] - sliceHeight
            if x0+sliceWidth > image0.shape[1]:
                x0 = image0.shape[1] - sliceWidth
        
            window_c = image0[y0:y0 + sliceHeight, x0:x0 + sliceWidth]

            new_name = name + '_' + str(x0) + '_' + str(y0) + '_' + str(sliceHeight) + '_' + str(sliceWidth) +'_' + '_' + str(win_w) + '_' + str(win_h) + ext
            cv2.imwrite(os.path.join(outdir,new_name), window_c)

# indir = 'raw/train/'
# outdir = 'train/JPEGImages/'

indir = 'raw/test/'
outdir = 'test/JPEGImages/'

imgs = os.listdir(indir)

for img in tqdm(imgs):
    img_path = os.path.join(indir, img)
    slice_im(img_path, outdir)
    
