import argparse

import torch
import torchvision

from dataset import GNHK, GNHKDataset
from model import Model
from utils import CTCLabelConverter, display_image, get_box_images, collate_fn, filter_boxes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(args):
    demo_path = 'demo_dataset/'
    gnhk = GNHK(demo_path)

    demo_dataset = GNHKDataset(gnhk.getDataFrame(), demo_path)
    demo_dataloader = torch.utils.data.DataLoader(demo_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # for (imgs, targets, texts) in demo_dataloader:
    #     display_image(imgs[0], targets[0]['boxes'], targets[0]['labels'], texts[0])

    # CRNN Model
    crnn_model = Model()
    print('loading pretrained model from %s' % args.saved_model)
    crnn_model.load_state_dict(torch.load(args.saved_model, map_location=device))

    # Converter
    converter = CTCLabelConverter()

    crnn_model.eval()
    with torch.no_grad():
        for (imgs, targets, texts) in demo_dataloader:
            # display_image(imgs[0], targets[0]['boxes'], targets[0]['labels'], texts[0])
            for idx, image in enumerate(imgs):
                bboxes = filter_boxes(targets[idx]['boxes'], targets[idx]['labels'])
                print(bboxes)
                image_boxes = get_box_images(image, bboxes)

                transform = torchvision.transforms.Resize((32,32))
                trans_list = []
                for i, img in enumerate(image_boxes):
                    # if targets[idx]['labels'][i] == 1:
                    trans_list.append(transform(img))
                img = torch.stack(trans_list).to(device)

                target_text = texts[idx] # This will later be used to compute CAR / WAR
                text, length = converter.encode(target_text)

                crnn_preds = crnn_model(img)
                preds_size = torch.IntTensor([crnn_preds.size(1)] * len(image_boxes))

                crnn_preds = crnn_preds.log_softmax(2)

                _, preds_index = crnn_preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)

                display_image(imgs[idx], bboxes, torch.IntTensor([1]*len(preds_str)), preds_str)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--saved_model', help="path to saved_model to evaluation", default='text-recognition.pth')

    args = parser.parse_args()
    demo(args)