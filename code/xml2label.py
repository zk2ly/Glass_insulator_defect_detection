import xml.etree.ElementTree as ET
import os

classes=['defect']


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('raw/test_annotations/%s.xml' %image_id)
    out_file = open('test/labels/%s.txt' %image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    first = True
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        if first:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]))
            first = False
        else:
            out_file.write('\n' + str(cls_id) + " " + " ".join([str(a) for a in bb]))
    in_file.close()
    out_file.close()


anns = os.listdir('raw/test_annotations/')

for ann in anns:
    image_id = ann.split('.')[0]
    convert_annotation(image_id)
