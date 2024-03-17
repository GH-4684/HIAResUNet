## Prerequisites:
1. Python 3.8
2. PyTorch 1.7.1
3. numpy
4. pillow
5. opencv
6. matplotlib
7. tqdm

## Dataset
Run `data/gen_dataset-colorjitter-smooth.ipynb` to generate the dataset.

## Training
```py
python train_ta-res18-unet-cml.py --epochs 100 --batch-size 4 --learning-rate 1e-5 --classes 1 --channels 2 --scale 0.5 --bilinear
```
The trained models can be found in `data`.

## Validation
```py
python evaluate2.py --model ./data/TAres18unet-pre-cml.pth --name TAResnet18_Unet --input_sar data/dataset/trainval_imgs/ --input_mask data/dataset/trainval_masks/ --output ./result_eval/out_ResUNet-TAM-CML.csv --classes 1 --channels 2 --scale 0.5 --bilinear --batch_size 4
```

## Prediction
```py
python predict.py --model ./data/TAres18unet-pre-cml.pth --name TAResnet18_Unet --input_sar data/demo/ --input_mask data/demo/ --output data/demo/output/ --classes 1 --channels 2 --scale 0.5 --bilinear
```

## Acknowledgement
This code is built on [U-Net: Semantic segmentation with PyTorch](https://github.com/milesial/Pytorch-UNet). We thank the authors for sharing their codes.
