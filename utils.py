import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np
import matplotlib.pyplot as plt
import cv2

import string
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Code from ClovaAI: https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/utils.py
class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        # character (str): set of the possible characters.
        character = string.printable[:-6]
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=50):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 50 by default
        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] if char != '£' else self.dict['L'] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[int(t[i])])
            text = ''.join(char_list)

            texts.append(text)
        return texts

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

def display_image(image, bboxes, labels, text):
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
        cv2.putText(np_img, text[idx], (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    ax.set_axis_off()
    ax.imshow(np_img)
    plt.show()

def get_box_images(image, boxes):
    img_list = []

    for x0, y0, x1, y1 in boxes:
        img_list.append(image[:, int(y0):int(y1), int(x0):int(x1)])

    return img_list

def display_images(img_list, texts):
    plt.figure(figsize=(16, 8))
    size = int(np.sqrt(len(img_list)))+1

    for idx, img in enumerate(img_list):
        np_img = img.permute(1,2,0).cpu().numpy().copy()
        
        plt.subplot(size, size, idx+1)
        plt.title(texts[idx])
        plt.axis('off')
        plt.imshow(np_img)
    
    plt.show()