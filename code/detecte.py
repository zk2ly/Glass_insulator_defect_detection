import os 
from insulator_data.split_images import slice_im
from insulator_data.yolo import YOLO
import numpy as np
import time
import cv2
import shutil
from tqdm import tqdm

def nms(dets, thresh=0.15):
    # 返回保留框的序号
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

images_dir = '/home/zk/darknet/insulator_data/raw/'
temp_dir = '/home/zk/darknet/insulator_data/temp/imgs/'
images = os.listdir(images_dir)

yolo = YOLO('/home/zk/darknet/cfg/yolov4-insulator.cfg', '/home/zk/darknet/backup/insulator/yolov4-insulator_best.weights', True)

for idx, image in enumerate(images):
    image_path = images_dir + image
    slice_im(image_path, temp_dir)

    labels_file = '/home/zk/darknet/insulator_data/temp/labels/'
    if not os.path.exists(labels_file):
        os.mkdir(labels_file)

    real_boxes = []
    print(idx+1, '/', 30)
    for temp_img in tqdm(os.listdir(temp_dir)):
        boxes = yolo.detect_cv2(os.path.join(temp_dir, temp_img))
        info = temp_img.split('.')[0].split('_')
        left = int(info[1])
        top = int(info[2])
        for x1, y1, x2, y2, conf in boxes:
            xmin = x1 + left
            xmax = x2 + left
            ymin = y1 + top
            ymax = y2 + top
            real_boxes.append([xmin, ymin, xmax, ymax, conf])
    real_boxes = np.asarray(real_boxes)
    nms_real_boxes = real_boxes[nms(real_boxes)]

    raw = cv2.imread(image_path)
    for box in nms_real_boxes:
        cv2.rectangle(raw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=2)
        cv2.putText(raw, '%.2f'%box[4], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), thickness=2)
        cv2.imwrite('/home/zk/darknet/insulator_data/predict/' + image, raw)
    shutil.rmtree(temp_dir)
    shutil.rmtree(labels_file)
