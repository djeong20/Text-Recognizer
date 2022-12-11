import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch

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


def main():
    gnhk = GNHK('gnhk/train')
    train_df = gnhk.getDataFrame()

    # print(train_df.head()) # Uncomment to see DataFrame


if __name__ == "__main__":
    main()
