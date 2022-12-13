
import numpy as np
import argparse
import tqdm

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import GNHK, GNHKDataset
from utils import Averager

def collate_fn(batch):
    return tuple(zip(*batch))

def train(args):
    train, validate = GNHK('gnhk/train'), GNHK('gnhk/test')

    train_dataset = GNHKDataset(train.getDataFrame(), 'gnhk/train/', A.Compose([A.Resize(height=512, width=512, p=1), ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}))
    val_dataset = GNHKDataset(validate.getDataFrame(), 'gnhk/test/')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    
    # Model: Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    output_features = 4 # %NA%, text, %math%, %SC%
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, output_features) 

    params = [p for p in model.parameters() if p.requires_grad]

    # Optimizers
    optimizer = torch.optim.Adam(params)

    # LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.5)
    loss_avg = Averager()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(args.n_epochs):
        model.train()
        loss_avg.reset()
        
        for (imgs, targets) in tqdm.tqdm(train_dataloader):
            imgs = torch.stack(imgs).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            preds = model(imgs, targets)
            
            cost = sum(loss for loss in preds.values())
            loss_avg.update(cost.item(), imgs.shape[0])

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        lr_scheduler.step(cost)

        print(f"Epoch {epoch+1}/{args.n_epochs}")
        print(f"Train loss: {loss_avg.val():0.5f}")
        loss_avg.reset()
        
        torch.save(model.state_dict(), 'text-recognition.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', dest="batch_size", type=int, default=16, help='input batch size. default=16')
    parser.add_argument('--n_epochs', dest="n_epochs", type=int, default=5, help='number of epoch. default=5')
    parser.add_argument('--workers', dest="workers", type=int, default=4, help='number of data loading workers. default=4')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    
    args = parser.parse_args()

    train(args)