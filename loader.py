import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import random

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

        # Load Image
        img = self.load_image(index)

        # Bounding Boxes
        boxes = data[['x0','y0','x1','y1','x2','y2','x3','y3']].values
        boxes = torch.as_tensor(boxes,dtype=torch.float32)

        target = {}
        target['image_id'] = torch.tensor([index])
        target['bboxes'] = boxes
        target['text'] = data.text.values # TODO: labels to integer or string

        return img, target, image_id

def displayImage(image, bboxes, image_id):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in bboxes:
        cv2.rectangle(image,
                      (min(box[0::2]), min(box[1::2])),
                      (max(box[0::2]), max(box[1::2])),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(image)
    ax.set_title(image_id)
    plt.show()

def main():
    gnhk = GNHK('gnhk/train')
    train_df = gnhk.getDataFrame()

    # print(train_df.head()) # Uncomment to see DataFrame
    train_dataset = GNHKDataset(train_df, 'gnhk/train/')
    
    # Display random sample dataset
    img, target, image_id = train_dataset[random.randint(0, len(train_dataset))]
    displayImage(img, target['bboxes'].numpy().astype(np.int32), image_id)


if __name__ == "__main__":
    main()
