from insulator_data.tool.utils import *
from insulator_data.tool.torch_utils import *
from insulator_data.tool.darknet2pytorch import Darknet
import cv2


class YOLO():
    def __init__(self, cfgfile, weightfile, use_cuda=False, gpu_id=None):
        print("\033[1;31;47mLoading YOLO model...\033[0m")
        self.m = Darknet(cfgfile)
        # self.m.print_network()
        self.m.load_weights(weightfile)
        self.class_names = load_class_names('/home/zk/darknet/data/insulator.names')
        self.use_cuda = use_cuda
        self.gpu_id = gpu_id
        if use_cuda:
            self.m.cuda(self.gpu_id)

    def detect_cv2(self, imgfile):
        img = cv2.imread(imgfile)
        imh, imw, _ = img.shape
        sized = cv2.resize(img, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        
        boxes = do_detect(self.m, sized, 0.5, 0.6, self.use_cuda, self.gpu_id)
        vehicle = []
        for idx, box in enumerate(boxes[0]):
            x1 = int(round(box[0] * imw))
            y1 = int(round(box[1] * imh))
            x2 = int(round(box[2] * imw))
            y2 = int(round(box[3] * imh))
            conf = box[4]
            vehicle.append([x1, y1, x2, y2, conf])
        return vehicle

if __name__ == '__main__':
    # yolo = YOLO('yolo/cfg/yolov4-custom.cfg', 'yolo/weight/yolov4-all.weights', True)
    # vehicle = yolo.detect_cv2('yolo/test.jpg')
    yolo = YOLO('/home/zk/darknet/cfg/yolov4-tiny-insulator.cfg', '/home/zk/darknet/backup/insulator/72.weights', True)
