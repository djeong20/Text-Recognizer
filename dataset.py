import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class GNHK():
    def __init__(self, path):
        self.path = path
    
    # Load JSON file to DataFrame
    def load_json(self):
        dir = sorted(os.listdir(self.path))
        df_list = []

        for name in dir:
            if '.json' in name:
                json = pd.read_json(os.path.join(self.path, name))

                # Get image id
                image_id = name.split('.')[0]
                json['image_id'] = image_id

                # Get source (AF, AS, EU, NA)
                source = image_id.split('_')[1]
                json['source'] = source

                df_list.append(json)
            
        df = pd.concat(df_list, axis=0)
        
        return df

    # Parse polygon column to x0, y1, ..., x3, y3
    def data_preprocess(self, df):
        def parsePolygon(x, key):
            return x.polygon[key]

        df['x0'] = df.apply(parsePolygon, axis=1, key='x0')
        df['y0'] = df.apply(parsePolygon, axis=1, key='y0')
        df['x1'] = df.apply(parsePolygon, axis=1, key='x1')
        df['y1'] = df.apply(parsePolygon, axis=1, key='y1')
        df['x2'] = df.apply(parsePolygon, axis=1, key='x2')
        df['y2'] = df.apply(parsePolygon, axis=1, key='y2')
        df['x3'] = df.apply(parsePolygon, axis=1, key='x3')
        df['y3'] = df.apply(parsePolygon, axis=1, key='y3')

        df = df.drop(['polygon'], axis=1)

        def convertLabel(x):
            tag = x.text

            # No char or math: 0, Text: 1; Math symbol: 2, Illegible scribbles: 3
            if tag == '%math%': return 2
            elif tag == '%SC%': return 3
            elif tag == '%NA%': return 0
            else: return 1

        df['label'] = df.apply(convertLabel, axis=1)

        return df

    def getDataFrame(self):
        df = self.load_json()
        df = self.data_preprocess(df)
        return df

class GNHKDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transforms=None):
        super().__init__()

        # Image_ids will be the "Filename" here
        self.image_ids = df.image_id.unique()
        self.sources = df.source.unique()

        self.root_dir = root_dir
        self.df = df
        self.transforms = transforms

    def load_image(self, index):
        image_id = self.image_ids[index]
        path = self.root_dir + image_id + '.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0

        return img
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.df[self.df.image_id == image_id]

        N = len(data)

        # Load Image
        img = self.load_image(index)

        # Labels
        labels = torch.as_tensor(data.label.values, dtype=torch.int64)
        
        # Bounding Boxes
        box = data[['x0','y0','x1','y1','x2','y2','x3','y3']].values
        boxes = []
        for i in range(N):
            pos = box[i]
            xmin = np.min(pos[0::2])
            xmax = np.max(pos[0::2])
            ymin = np.min(pos[1::2])
            ymax = np.max(pos[1::2])
            boxes.append([xmin, ymin, xmax, ymax])

        # boxes = data[['x0','y0','x1','y1','x2','y2','x3','y3']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # All instances are not crowd
        iscrowd = torch.zeros((N,), dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([index])
        target["labels"] = labels
        target['boxes'] = boxes
        target["area"] = area
        target["iscrowd"] = iscrowd
        # target['text'] = data.text.values # TODO

        return img, target

# Background=Blue, Text=Red, Math=Green, Scribbles=Yellow
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]

def displayImage(image, bboxes, labels):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for idx, box in enumerate(bboxes):
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      colors[labels[idx]], 3)

    ax.set_axis_off()
    ax.imshow(image)
    plt.show()

def collate_fn(batch):
    return tuple(zip(*batch))

def verify_dataset():
    train = GNHK('gnhk/train')
    train_dataset = GNHKDataset(train.getDataFrame(), 'gnhk/train/')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # Display train dataset
    imgs, targets = next(iter(train_dataloader))
    displayImage(imgs[0], targets[0]['boxes'].numpy().astype(np.int32), targets[0]['labels'].numpy().astype(np.int32))


if __name__ == "__main__":
    verify_dataset()
