# Text-Recognizer

## Getting Started
### Dependency
- requirements : numpy, pandas, matplotlib, pytorch, opencv-python, torchvision, albumentations, and Levenshtein
```
pip3 install numpy pandas tqdm matplotlib opencv-python torch torchvision albumentations Levenshtein
```

### Download GNHK dataset for training from [here](https://www.goodnotes.com/gnhk)

### Folder structure

The structure of folder as below.

```
Text-Recognizer
├── README.md
├── dataset.py
├── model.py
├── train.py
├── utils.py
└── gnhk
    ├── test
    └── train
```

### Check Data Pipeline

```
 python3 dataset.py --data='gnhk/train/' --transform=False 
```
#### Arguments
* `--data`: folder path to gnhk dataset.
* `--transform`: transformation. (default=False)


### Training

```
 python3 train.py --train_data='gnhk/train/' --valid_data='gnhk/test/' \
 --train_trans=True --valid_trans=True \
 --batch_size=16 --n_epochs=5 --grad_clip=0.5 --workers=4
```
#### Arguments
* `--train_data`: folder path to gnhk training dataset.
* `--valid_data`: folder path to gnhk validation dataset.
* `--train_trans`: transformation in training. (default=True)
* `--valid_trans`: transformation in validation. (default=True)
* `--batch_size`: input batch size. (default=16)
* `--n_epochs`: number of epoches. (default=5)
* `--grad_clip`: gradient clipping value. (default=0.5)
* `--workers`: number of data loading workers. (default=4)






