import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
import cv2

class Averager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_train_transform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_validate_transform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def display_image(image, bboxes, labels):
    # Background=Blue, Text=Red, Math=Green, Scribbles=Yellow
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    np_img = image.permute(1,2,0).cpu().numpy().copy()
    np_boxes = bboxes.cpu().numpy().astype(np.int32)
    np_labels = labels.numpy().astype(np.int32)
    
    for idx, box in enumerate(np_boxes):
        cv2.rectangle(np_img,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      colors[np_labels[idx]], 3)

    ax.set_axis_off()
    ax.imshow(np_img)
    plt.show()
