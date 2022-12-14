
import numpy as np
import argparse
import tqdm

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import GNHK, GNHKDataset
from utils import Averager, CTCLabelConverter
from utils import get_train_transform, get_validate_transform, get_box_images
from model import Model

def collate_fn(batch):
    return tuple(zip(*batch))

def train(args):
    train, validate = GNHK(args.train_data), GNHK(args.valid_data)

    train_transform = get_train_transform() if args.train_trans else None
    train_dataset = GNHKDataset(train.getDataFrame(), args.train_data, transforms=train_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    valid_transform = get_validate_transform() if args.valid_trans else None
    val_dataset = GNHKDataset(validate.getDataFrame(), args.valid_data, transforms=valid_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    
    # Model: Faster R-CNN / TODO: Replace with our own model
    rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    input_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
    output_features = 4 # %NA%, text, %math%, %SC%
    rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(input_features, output_features) 

    params = [p for p in rcnn_model.parameters() if p.requires_grad]

    # Optimizers
    optimizer = torch.optim.Adam(params)


    # LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.5)
    train_loss_avg = Averager()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model: CRNN 
    converter = CTCLabelConverter()
    crnn_model = Model()

    # Optimizers
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, crnn_model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    rec_optimizer = torch.optim.Adam(filtered_parameters, lr=0.001, betas=(0.9, 0.999))

    for name, param in crnn_model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    for epoch in range(args.n_epochs):
        rcnn_model.train()
        train_loss_avg.reset()
        crnn_model.train()
        
        # train for one epoch
        for (imgs, targets, texts) in tqdm.tqdm(train_dataloader):
            imgs = torch.stack(imgs).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            """ 
            TODO: Replace with our own model (Sanghyuk Seo)
            -- Need our model that outputs label and bounding boxes 
            
            e.g. labels, bboxes = rcnn_model(imgs, targets)

            -> Use labels to compute accuracy
            -> Use bboxes to compute IOU
            -> Use both to compute recall, precision, f-measure

            """
            preds = rcnn_model(imgs, targets)
            
            cost = sum(loss for loss in preds.values())
            train_loss_avg.update(cost.item(), imgs.shape[0])

            rcnn_model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(rcnn_model.parameters(), args.grad_clip)
            optimizer.step()

            # crnn
            for idx, image in enumerate(imgs):
                img_list = get_box_images(image, targets[idx]['boxes'])
                transform = torchvision.transforms.Resize((32,32))
                trans_list = []
                for img in img_list:
                    trans_list.append(transform(img))

                img = torch.stack(trans_list).to(device)

                target_text = texts[idx] # This will later be used to compute CAR / WAR
                text, length = converter.encode(target_text)

                preds2 = crnn_model(img)
                preds_size = torch.IntTensor([preds2.size(1)] * len(img_list))

                preds2 = preds2.log_softmax(2)

                # TODO: 1. Need to decode preds2 to texts to compute CAR/WAR (TEMP: Need to perform this in evaluation section)

                crnn_cost = criterion(preds2.permute(1, 0, 2), text, preds_size, length)

                # TODO: 2. After 1, compute CAR / WAR using target_text (TEMP: Need to perform this in evaluation section)

                crnn_model.zero_grad()
                crnn_cost.backward()
                torch.nn.utils.clip_grad_norm_(crnn_model.parameters(), args.grad_clip)  # gradient clipping with 5 (Default)
                rec_optimizer.step()
        
        print(f"Epoch {epoch+1}/{args.n_epochs}")
        print(f"Train loss: {train_loss_avg.val():0.5f}")
        
        # update the learning rate
        lr_scheduler.step(cost)

        """ 
        TODO: Evaluate both localization and recognition using Testing dataset 
        
        Localization evaluation: Sanghyuk Seo
        Expected results: cost, IOU, accuracy, recall, precision, F-score

        Recognition evalution: Eunjeong Ro 
        Expected results: cost, CAR, WAR (expected to be around 0)

        """

        train_loss_avg.reset()
        
        torch.save(rcnn_model.state_dict(), 'text-recognition.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, default='gnhk/train/', help='path to gnhk training dataset')
    parser.add_argument('--valid_data', type=str, default='gnhk/test/', help='path to gnhk validation dataset')
    parser.add_argument('--train_trans', type=bool, default=True, help='transformation in training')
    parser.add_argument('--valid_trans', type=bool, default=True, help='transformation in validation')
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=16, help='input batch size. default=16')
    parser.add_argument('--n_epochs', dest="n_epochs", type=int, default=5, help='number of epoch. default=5')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--workers', dest="workers", type=int, default=4, help='number of data loading workers. default=4')
    
    args = parser.parse_args()

    train(args)