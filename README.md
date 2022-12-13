# Text-Recognizer

## Getting Started
### Dependency
- requirements : numpy, pandas, matplotlib, pytorch, opencv-python, torchvision
```
pip3 install numpy pandas matplotlib opencv-python torch torchvision
```

### Download GNHK dataset for training from [here](https://www.goodnotes.com/gnhk)

### Folder structure

The structure of folder as below.

```
Text-Recognizer
├── README.md
├── dataset.py
├── train.py
├── demo.py
├── utils.py
└── gnhk
    ├── test
    └── train
```

### Training

```
 python3 train.py \
--batch_size BATCH_SIZE \
--n_epochs N_EPOCHS \
--grad_clip GRAD_CLIP \
--workers WORKERS
```

### Arguments
* `--batch_size`: input batch size
* `--n_epochs`: number of epoches
* `--grad_clip`: gradient clipping value
* `--workers`: number of data loading workers
