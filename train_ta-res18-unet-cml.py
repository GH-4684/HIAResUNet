import argparse
import logging
import os
import random
import re
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import shutil
from PIL import Image
import numpy as np
from utils.utils import plot_and_save_pred
from utils.dice_score import dice_coeff, multiclass_dice_coeff
import wandb
from evaluate import evaluate
from unet.TARes18UNet import TAResnet18_Unet
from utils.data_loading import BasicDataset
from utils.dice_score import cml_loss
from utils.utils import save_feature_map, plot_save_loss


# 可视化配置
using_wandb = True
wandb_project = 'unet-2k-ss'
wandb_name = 'ta-res18-unet-pre-cml-untitled'
remove_checkpoints = False           # 自动删除上一次的checkpoints
feature_map_index_to_save = []    # 选择第几张测试集图片进行特征图分析
# 训练参数配置
beta = 0.5
weight_decay = 1e-8
momentum = 0.99
# 输入路径
dir_img = [Path('./data/dataset/trainval_imgs/')]
dir_mask = [Path('./data/dataset/trainval_masks/')]
dir_test_img = [Path('./data/dataset/test_imgs')]
dir_test_mask = [Path('./data/dataset/test_masks')]
# mask像素所有取值
mask_values = [np.array((0,), dtype='u1'), np.array((255,), dtype='u1')]
# 输出路径
dir_checkpoint = Path('./data/checkpoints/')
dir_test_output = Path('./data/checkpoints/')
dir_feature_map_output = Path('./data/checkpoints/feature_map')


def train_model(
        model: TAResnet18_Unet,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        # weight_decay: float = 1e-8,
        # momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, model.n_channels, img_scale, mask_values=mask_values)
    testset = BasicDataset(dir_test_img, dir_test_mask, model.n_channels, img_scale, mask_values=mask_values)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # num_workers = os.cpu_count()
    num_workers = 0
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    if using_wandb:
        experiment = wandb.init(project=wandb_project, name=wandb_name, resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)  # goal: maximize Dice score
    
    # Warm up + Cosine Anneal
    # T_max = epochs * len(train_loader)
    # warm_up_iter = T_max // 10
    # lr_max = 1
    # lr_min = 1e-4
    # lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
    #     (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    epoch_start = 1
    global_step = 0

    if data:
        epoch_start = int(data[-1][0]) + 1
        global_step = int(data[-1][1])

    # 5. Begin training
    for epoch in range(epoch_start, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = model(images)
                    loss = cml_loss(model, masks_pred, true_masks, model.n_classes, beta=beta, K=0.3)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # scheduler.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                data.append((epoch, global_step, loss.item(), None, None, None, None))
                if using_wandb:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        if using_wandb:
                            histograms = {}
                            # for tag, value in model.named_parameters():
                            #     tag = tag.replace('/', '.')
                            #     if not torch.isinf(value).any():
                            #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            #     if not torch.isinf(value.grad).any():
                            #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_loss = evaluate(model, val_loader, device, amp, beta=beta)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        data.append((epoch, global_step, None, val_loss.item(), None, val_score.item(), None))
                        if using_wandb:
                            image = images[0, 0].cpu()
                            true = true_masks[0].float().cpu()
                            if model.n_classes == 1:
                                pred = masks_pred[0, 0].float().cpu()
                            else:
                                pred = masks_pred.argmax(dim=1)[0].float().cpu()
                            pred = torch.sigmoid(pred)
                            wandb_image = wandb.Image(torch.hstack((image, true, pred)))
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'validation loss': val_loss,
                                    'image&true&pred': wandb_image,
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass

        if save_checkpoint:
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch:02d}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

            plot_save_loss(data, savedir=Path(dir_checkpoint), save_fig=True)

            test_dice_score = 0
            test_loss = 0
            model.eval()
            for i,batch in enumerate(tqdm(test_loader, total=len(test_loader), desc='Test round', unit='batch', leave=True)):
                if i in feature_map_index_to_save:
                    model.register_forward_hooks()

                image, true_mask = batch['image'], batch['mask']
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_mask = true_mask.to(device=device, dtype=torch.long)
                with torch.no_grad():
                    mask_pred =  model(image)
                test_loss += cml_loss(model, mask_pred, true_mask, model.n_classes, beta=beta, K=0.3)
                if model.n_classes > 1:
                    mask_pred = mask_pred.argmax(dim=1) # [1,H,W]
                else:
                    mask_pred = torch.sigmoid(mask_pred).squeeze(0) > 0.5
                dice = dice_coeff(mask_pred, true_mask)
                test_dice_score += dice

                save_path = dir_checkpoint / f'checkpoint_epoch{epoch:02d}_test{i:04d}_dice={dice:.4f}.png'
                image, mask_pred, true_mask = image.cpu(), mask_pred.cpu(), true_mask.cpu()
                Image.fromarray((image[0,0].numpy()*255).astype('u1')).save(Path(save_path).with_name(f'test{i:04d}.png'))
                plot_and_save_pred(None, mask_pred.squeeze(0).numpy(), true_mask.squeeze(0).numpy(), save_path, viz=False, methods=('diff_rgb'))

                if i in feature_map_index_to_save:
                    model.close_forward_hooks()

                    savepath = dir_feature_map_output/f'{i:04d}_input.png'
                    save_feature_map(image, savepath)

                    for n, (name, feature_map) in enumerate(model.feature_maps):
                        savepath = dir_feature_map_output/f'{i:04d}_mod{n:02d}={name}_epoch{epoch:02d}.png'
                        save_feature_map(feature_map, savepath)
                        savepath = dir_feature_map_output/f'{i:04d}_mod{n:02d}={name}_fusion_epoch{epoch:02d}.png'
                        feature_map = feature_map.mean(1, keepdim=True)
                        save_feature_map(feature_map, savepath)
                    model.clear_feature_maps()

            model.train()
            test_dice_score = test_dice_score / len(test_loader)
            test_loss = test_loss / len(test_loader)
            logging.info('Test Dice score: {}'.format(test_dice_score))
            data.append((epoch, global_step, None, None, test_loss.item(), None, test_dice_score.item()))
            if using_wandb:
                try:
                    experiment.log({
                        'test Dice': test_dice_score,
                        'test loss': test_loss,
                        'step': global_step,
                        'epoch': epoch,
                    })
                except:
                    pass


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels')

    return parser.parse_args()


if __name__ == '__main__':
    if remove_checkpoints:
        shutil.rmtree(dir_checkpoint, ignore_errors=True)
        shutil.rmtree(dir_test_output, ignore_errors=True)
        shutil.rmtree(dir_feature_map_output, ignore_errors=True)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    Path(dir_test_output).mkdir(parents=True, exist_ok=True)
    Path(dir_feature_map_output).mkdir(parents=True, exist_ok=True)

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    seed = 0
    torch.manual_seed(seed)

    # Change here to adapt to your data
    model = TAResnet18_Unet(n_channels=args.channels, n_classes=args.classes, pretrained=True)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    data = []   # (epoch, global_step, train_loss, val_loss, test_loss, val_dice, test_dice)
    model_weight = None

    if args.load:
        if Path(args.load).is_dir():
            temp = re.findall(f'checkpoint_epoch(\d+).pth', ' '.join(os.listdir(args.load)))
            if temp:
                epoch = max([int(x) for x in temp])
                data = np.load(Path(args.load)/f'data_epoch{epoch:02.0f}.npy').tolist()
                model_weight = Path(args.load)/f'checkpoint_epoch{epoch:02.0f}.pth'
        elif Path(args.load).is_file():
            model_weight = args.load
        else:
            raise Exception(f'error: {args.load}')

        if model_weight:
            state_dict = torch.load(model_weight, map_location=device)
            if mask_values is None:
                mask_values = state_dict['mask_values']
            del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {model_weight}')

    model.to(device=device)

    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp
    )

    plot_save_loss(data, savedir=Path(dir_checkpoint))
