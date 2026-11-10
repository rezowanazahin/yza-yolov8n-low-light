# Y-ZA YOLOv8n: low-light object detection

YOLOv8n with Efficient Channel Attention (ECA) and Global Context (GC) modules added after each of the four C2f layers in the neck.

This is a re-implementation written from the paper's description. Details such as the order of ECA and GC, the GC reduction ratio, and the training seed may differ from the original experiments, so results can differ slightly from the paper.

## Files
- `attention_modules.py`: the ECA and GC modules, and `add_attention` which inserts them.
- `train_yza.py`: trains Y-ZA YOLOv8n.
- `train_baseline.py`: trains plain YOLOv8n with the same settings.
- `check_model.py`: builds both models, prints parameter counts, runs one forward pass.
- `data.yaml`: dataset configuration (edit the path).

## Setup
```bash
pip install ultralytics
```

## Data
Download ExDark from the official page (https://github.com/cs-chan/Exclusively-Dark-Image-Dataset), which gives `ExDark/` (images) and `ExDark_Anno/` (annotations). Convert it to YOLO format (12 classes: Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table):
```bash
python convert_exdark.py --src /path/to/folder_with_ExDark --dst datasets/ExDark
```
Then set `path` in `data.yaml` to the **absolute** path of `datasets/ExDark`. If you export the dataset from Roboflow instead, use the `data.yaml` it provides.

## Google Colab
`YZA_YOLOv8n_Colab.ipynb` runs the whole pipeline on a free GPU: convert the data, train both models, and print both mAP@0.5 scores.

## Run
```bash
python check_model.py     # sanity check
python train_baseline.py  # baseline, 50 epochs, batch 8
python train_yza.py       # Y-ZA YOLOv8n, 50 epochs, batch 8
```

Results (mAP@0.5, precision, recall, curves) are saved under `runs/`.

## Results
Measured on the ExDark validation split (1,799 images), trained from scratch for 50 epochs with batch size 8 on a Tesla T4 (Google Colab). A single run each; results vary slightly between runs.

| Model | Epochs | Batch | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8n (baseline) | 50 | 8 | 0.404 | 0.202|

## References
- Wang et al., ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, CVPR 2020.
- Cao et al., GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond, ICCV Workshops 2019.
- Loh and Chan, Getting to Know Low-light Images with the Exclusively Dark Dataset, CVIU 2019.
