import imageio
import imgaug as ia
import cv2
from PIL import Image
import numpy as np
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
import pybboxes as pbx
import argparse

PATH_IMAGES = "data/images"
PATH_LABELS = "data/labels"

#TODO:Add destination folder 
#TODO: Add folder checks, right or wrong folders
#TODO: Add ReadMe
#TODO: Add bouding box errors after augmentations

def get_files(path:str, ext:str):
    return [os.path.join(path,file) for file in os.listdir(path) if file.endswith(ext)]


def load_image(path:str):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def get_the_bbs_from_image(img_name:str, path_to_labels:str):
    filename = os.path.splitext(os.path.basename(img_name))[0] + '.txt'
    with open(os.path.join(path_to_labels, filename), 'r') as f:
        #rstrip() takes individual lines in file without final \n, split() splits at every space
        list_bounding_boxes= [line.rstrip().split() for line in f]
    f.close()
    #two for loops: as list[list]; check for float or int for each item in inside list based on '.'
    return [[float(i) if '.' in i else int(i) for i in item] for item in list_bounding_boxes]

def get_img_shape(path:str):
    im = cv2.imread(path)
    return im.shape

        

def get_bbs_augment_object(list_bbs:list, img_shape:tuple):

# https://nbviewer.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb
# https://stackoverflow.com/questions/62459621/how-to-batch-process-with-multiple-bounding-boxes-in-imgaug
# https://github.com/devrimcavusoglu/pybboxes 
# https://imgaug.readthedocs.io/en/latest/source/jupyter_notebooks.html 

# voc : Pascal VOC : [x-tl, y-tl, x-br, y-br] Top-left coordinates & Bottom-right coordinates
# yolo : YOLO : [x-c, y-c, w, h] Center coordinates & width & height


    def create_BoundingBox(bbox, w, h):
        bbox_mod = pbx.convert_bbox(bbox[1:],"yolo", "voc", image_size= (int(h), int(w)))
        return BoundingBox(bbox_mod[1], bbox_mod[0], bbox_mod[3], bbox_mod[2], bbox[0])
        #return BoundingBox(bbox[1]*h, bbox[2]*w, bbox[3]*h, bbox[4]*w, bbox[0])
    
    
    return BoundingBoxesOnImage([create_BoundingBox(bbox, float(img_shape[1]), float(img_shape[0]))for bbox in list_bbs], img_shape)


def write_augmented_label_file(bbs_aug, filename):
    bbs_res = []
    for item in bbs_aug.items:
        #BoundingBox(bbox_mod[1], bbox_mod[0], bbox_mod[3], bbox_mod[2], bbox[0])
        x_tl= item.y1
        y_tl= item.x1
        x_br= item.y2
        y_br= item.x2
        label = item.label
        yolo_boxes = [label] + list(pbx.convert_bbox((x_tl, y_tl, x_br,y_br), "voc", "yolo", image_size=(bbs_aug.shape[:2])))
        bbs_res.append(yolo_boxes)
    with open(filename+'.txt', 'w') as f:
        for item in bbs_res:
            for value in item:
                f.write(str(value)+ ' ')
            f.write('\n')
    f.close()
        
def write_aug_image(img, filename):
    im = Image.fromarray(img)
    im.save(filename)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=PATH_IMAGES, help= 'Path to source images')
    parser.add_argument('--label_path', type=str, default=PATH_LABELS, help='Path to the original labels')
    parser.add_argument('--image_extn', type=str, default='.jpg', help='Image file extensions')
    parser.add_argument('--target_folder', type=str, default='.', help='Target folder to save the augmented images and labels')
    parser.add_argument('--prefix', type=str, default="augment_", help="prefix string to add to the augmented files")
    opt = parser.parse_args()

    ia.seed(1)

    seq = iaa.Sequential([
    iaa.GammaContrast(1.5),
    iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)
        ])
    
    images_list = get_files(opt.image_path, opt.image_extn)
    for image in images_list:
        bbs_list= get_the_bbs_from_image(image, opt.label_path)
        img_shape = get_img_shape(image)
        image_aug, bbs_aug = seq(image=load_image(image), bounding_boxes=get_bbs_augment_object(bbs_list, img_shape))
        write_augmented_label_file(bbs_aug, opt.prefix + os.path.splitext(os.path.basename(image))[0])
        write_aug_image(image_aug, opt.prefix + os.path.basename(image))
    
    print("End of Augmentation, Totally {} images augmented".format(len(images_list)))


