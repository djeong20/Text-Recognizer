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
import Levenshtein

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

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """ Faster R-CNN """
    # Model: Faster R-CNN / TODO: Replace with our own model
    rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    input_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
    output_features = 4 # %NA%, text, %math%, %SC%
    rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(input_features, output_features) 

    # Optimizers (Faster RCNN)
    params = [p for p in rcnn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)

    # LR Scheduler (Faster RCNN)
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.5)
    rcnn_train_loss_avg = Averager()
    rcnn_valid_loss_avg = Averager()

    """ CRNN """
    # Model: CRNN 
    converter = CTCLabelConverter()
    crnn_model = Model()

    # Optimizers (CRNN)
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, crnn_model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    rec_optimizer = torch.optim.Adam(filtered_parameters, lr=0.001, betas=(0.9, 0.999))

    # LR Scheduler (CRNN)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.5)
    crnn_train_loss_avg = Averager()
    crnn_valid_loss_avg = Averager()

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
        rcnn_train_loss_avg.reset()
        
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
            
            rcnn_cost = sum(loss for loss in preds.values())
            rcnn_train_loss_avg.update(rcnn_cost.item(), imgs.shape[0])

            rcnn_model.zero_grad()
            rcnn_cost.backward()
            torch.nn.utils.clip_grad_norm_(rcnn_model.parameters(), args.grad_clip)
            optimizer.step()

            # CRNN part
            crnn_model.train()
            crnn_train_loss_avg.reset()

            for idx, image in enumerate(imgs):
                image_boxes = get_box_images(image, targets[idx]['boxes'])
                transform = torchvision.transforms.Resize((32,32))
                trans_list = []
                for img in image_boxes:
                    trans_list.append(transform(img))

                img = torch.stack(trans_list).to(device)

                target_text = texts[idx] # This will later be used to compute CAR / WAR
                text, length = converter.encode(target_text)

                crnn_preds = crnn_model(img)
                preds_size = torch.IntTensor([crnn_preds.size(1)] * len(image_boxes))

                crnn_cost = criterion(crnn_preds.log_softmax(2).permute(1, 0, 2), text, preds_size, length)
                crnn_train_loss_avg.update(crnn_cost.item(), len(image_boxes))

                crnn_model.zero_grad()
                crnn_cost.backward()
                torch.nn.utils.clip_grad_norm_(crnn_model.parameters(), args.grad_clip)  # gradient clipping with 5 (Default)
                rec_optimizer.step()

            # update the learning rate for CRNN
            lr_scheduler2.step(crnn_cost)
            
            # Uncomment to check evaluation faster
            # break 

        print(f"Epoch {epoch+1}/{args.n_epochs}")
        print(f"Train loss (Faster R-CNN): {rcnn_train_loss_avg.val:0.5f}")
        print(f"Train loss (CRNN): {crnn_train_loss_avg.val:0.5f}")
        
        # update the learning rate for Faster R-CNN
        lr_scheduler1.step(rcnn_cost)


        """ 
        TODO: Evaluate both localization using Testing dataset 
        
        Localization evaluation: Sanghyuk Seo
        Expected results: cost, IOU, accuracy, recall, precision, F-score

        Recognition evalution: Eunjeong Ro (DONE)
        Expected results: cost, CAR, WAR (expected to be around 0)

        """

        print("Evaluating models...")
        crnn_model.eval()

        with torch.no_grad():
            # 1. Load validation dataset
            
            rcnn_valid_loss_avg.reset() # TODO
            crnn_valid_loss_avg.reset()

            for (imgs, targets, texts) in val_dataloader:
                # TODO: Evaluate Faster R-CNN

                # Evaluate CRNN using test dataset
                for idx, image in enumerate(imgs):
                    image_boxes = get_box_images(image, targets[idx]['boxes'])
                    transform = torchvision.transforms.Resize((32,32))
                    trans_list = []
                    for img in image_boxes:
                        trans_list.append(transform(img))

                    img = torch.stack(trans_list).to(device)

                    target_text = texts[idx] # This will later be used to compute CAR / WAR
                    text, length = converter.encode(target_text)

                    crnn_preds = crnn_model(img)
                    preds_size = torch.IntTensor([crnn_preds.size(1)] * len(image_boxes))

                    crnn_preds = crnn_preds.log_softmax(2)

                    crnn_cost = criterion(crnn_preds.permute(1, 0, 2), text, preds_size, length)
                    crnn_valid_loss_avg.update(crnn_cost.item(), len(image_boxes))

                    _, preds_index = crnn_preds.max(2)
                    preds_str = converter.decode(preds_index.data, preds_size.data)
                    
                    car = 0
                    distance = 0
                    for pred, target in zip(preds_str, target_text):
                        edit_distance = Levenshtein.distance(pred, target)
                        distance = 1 - edit_distance / max(len(pred), len(target))
                        car += distance
                        
                    car /= len(target_text)
                                    
                    war = 0
                    for pred, target in zip(preds_str, target_text):
                        if pred == target:
                            war += 1
                    war /= len(target_text)

            # print(f"Valid loss (Faster R-CNN): {rcnn_valid_loss_avg.val:0.5f}") # TODO
            print(f"Test loss (CRNN): {crnn_valid_loss_avg.val:0.5f}")
            print(f"CAR: {car:0.5f}")
            print(f"WAR: {war:0.5f}")

        
        torch.save(rcnn_model.state_dict(), 'text-localization.pth')
        torch.save(crnn_model.state_dict(), 'text-recognition.pth')

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