import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import dice_crossentropy_loss

from pathlib import Path
import numpy as np
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader, random_split
import unet
import json
import surface_distance as surfdist
import pandas as pd
import time
import argparse
import logging


class Score:
    def __init__(self, pred_mask: np.ndarray=None, true_mask: np.ndarray=None, classes: int=1, score_dict={}) -> None:
        '''输入[H,W]或[N,H,w], pred_mask为预测概率图0-1'''
        if pred_mask is None and true_mask is None:
            self.score = score_dict # 无图像输入，只输入score_dict
            return

        assert pred_mask.shape == true_mask.shape
        self.H, self.W = true_mask.shape
        self.num = self.H * self.W
        self.pred_mask = pred_mask
        self.true_mask = true_mask

        self.cm = self.get_cm()

        self.surface_distances = self.get_sd()

        self.classes = classes

        score_dict.update(dict(   # 输入图像和score_dict，作为额外参数
            P=self.precision,
            R=self.recall,
            ACC=self.accuracy,
            F1=self.F1,
            IoU=self.IoU,
            Dice=self.dice,
            Kappa=self.kappa,
            ASSD=self.assd,
            HD95=self.hd_95,
            HD100=self.hd_100,
            MAE=self.MAE,
        ))
        self.score = score_dict

    @property
    def precision(self):
        return (self.cm[0,0] + 1e-8) / (self.cm[:,0].sum() + 1e-8)
    @property
    def recall(self):
        return (self.cm[0,0] + 1e-8) / (self.cm[0,:].sum() + 1e-8)
    @property
    def accuracy(self):
        return self.cm.diagonal().sum() / self.cm.sum()
    @property
    def F1(self):
        P = self.precision
        R = self.recall
        return 2 * P*R/(P+R+1e-8)
    @property
    def IoU(self):
        return (self.cm[0,0] + 1e-8) / (self.cm.sum() - self.cm[1,1] + 1e-8)
    @property
    def dice(self):
        return (2 * self.cm[0,0] + 1e-8) / (2 * self.cm[0,0] + self.cm[0,1] + self.cm[1,0] + 1e-8)
    @property
    def kappa(self):
        p_o = self.accuracy
        p_e = (self.cm.sum(axis=0) @ self.cm.sum(axis=1)) / self.cm.sum()**2
        return (p_o - p_e) / (1 - p_e + 1e-8)

    @property
    def assd(self):
        avg_surf_dist = surfdist.compute_average_surface_distance(self.surface_distances)
        avg_surf_dist = sum(avg_surf_dist) / 2
        if np.isnan(avg_surf_dist):
            avg_surf_dist = 0
        return avg_surf_dist
    @property
    def hd_95(self):
        hd_dist_95 = surfdist.compute_robust_hausdorff(self.surface_distances, 95)
        if np.isinf(hd_dist_95):
            hd_dist_95 = 0
        return hd_dist_95
    @property
    def hd_100(self):
        hd_dist_100 = surfdist.compute_robust_hausdorff(self.surface_distances, 100)
        if np.isinf(hd_dist_100):
            hd_dist_100 = 0
        return hd_dist_100
    @property
    def MAE(self):
        mae = np.abs(self.pred_mask-self.true_mask).mean()
        return mae

    def get_cm(self, thres=0.5):
        pred_mask_ = self.pred_mask > thres

        if pred_mask_.ndim == 2 and self.true_mask.ndim == 2 and classes == 1:    # [H,W]
            pred_mask_ = np.stack((pred_mask_, 1-pred_mask_), axis=2)
            true_mask_ = np.stack((self.true_mask, 1-self.true_mask), axis=2)
        else:
            raise Exception()
        _, _, N = true_mask_.shape
        cm = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                cm[i, j] = (true_mask_[:,:,i] * pred_mask_[:,:,j]).sum()
        return cm

    def get_sd(self, thres=0.5):
        pred_mask_ = self.pred_mask > thres
        surface_distances = surfdist.compute_surface_distances(
            pred_mask_.astype(np.bool_), self.true_mask.astype(np.bool_), spacing_mm=(1.0, 1.0))
        return surface_distances

    def get_pr(self, thres):
        cm = self.get_cm(thres)
        precision = (cm[0,0] + 1e-8) / (cm[:,0].sum() + 1e-8)
        recall = (cm[0,0] + 1e-8) / (cm[0,:].sum() + 1e-8)
        return precision, recall

    def get_pr_data(self) -> dict:
        pr_data = []
        thresholds = np.linspace(0, 1, 1001)
        # thresholds = np.logspace(-8, 0, 101)
        for th in thresholds:
            pr_data.append(self.get_pr(th))
        pr_data = np.array(pr_data)
        return pr_data

    def get_score(self) -> dict:
        return self.score

    def append(self, other):
        if not self.score:
            return other
        score1 = self.get_score()
        score2 = other.get_score()
        score_new = dict()
        for k,v in score1.items():
            if type(v) != list:
                score_new[k] = [v]
            else:
                score_new[k] = v
            score_new[k].append(score2[k])
        self.score = score_new
        return self
    
    def calc_stat(self):
        score1 = self.get_score()
        score_mean = dict()
        score_max = dict()
        score_min = dict()
        for k,v in score1.items():
            if type(v[0]) == str:
                score_mean[k] = 'MEAN'
                score_max[k] = 'MAX'
                score_min[k] = 'MIN'
            else:
                score_mean[k] = np.mean(v)
                score_max[k] = np.max(v)
                score_min[k] = np.min(v)
        self.append(Score(score_dict=score_min))
        self.append(Score(score_dict=score_max))
        self.append(Score(score_dict=score_mean))
        return score_mean


@torch.no_grad()
def evaluate(net, dataloader, device, amp, beta):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0
    score = Score()
    pr_data_list = []
    # dice_score = 0
    # dist_score = 0

    # iterate over the validation set
    with torch.cuda.amp.autocast(enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            time1 = time.perf_counter()

            # predict the mask
            with torch.no_grad():
                mask_pred = net(image)  # 返回值的requires_grad=False
            
            dt = time.perf_counter() - time1

            val_loss += dice_crossentropy_loss(net, mask_pred, mask_true, net.n_classes, beta=beta)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = torch.sigmoid(mask_pred)
                # # compute the Dice score
                # dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
                # temp = calc_ASSD_HD95(mask_pred.squeeze(1), mask_true)
                # if not torch.isnan(temp).any():
                #     dist_score += temp
                for i in range(mask_pred.shape[0]):
                    score_ = Score(mask_pred.squeeze(1)[i].cpu().numpy(),
                                   mask_true[i].cpu().numpy(),
                                   score_dict=dict(name=batch['name'][i], dt=dt, fps=1/dt))
                    pr_data = score_.get_pr_data()
                    pr_data_list.append(pr_data)
                    score = score.append(score_)
            else:
                raise Exception()

    pr_data_ave = np.array(pr_data_list).mean(axis=0)

    net.train()
    # val_loss = val_loss.item() / max(num_val_batches, 1)
    # dice_score = dice_score.item() / max(num_val_batches, 1)
    # dist_score = (dist_score / max(num_val_batches, 1)).tolist()
    # return val_loss, dice_score, dist_score
    return score, pr_data_ave


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--name', help='Model name', required=True)
    parser.add_argument('--input_sar', nargs='+', help='Filenames of input sar images', required=True)
    parser.add_argument('--input_mask', nargs='+', help='Filenames of input mask images')
    parser.add_argument('--output', default='./result_eval/out.csv', help='export to csv file')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', type=int, default=3, help='Number of channels')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 参数设置
    args = get_args()
    assert args.input_sar and args.input_mask
    dir_img = args.input_sar
    dir_mask = args.input_mask
    save_path = Path(args.output)
    channels = args.channels
    classes = args.classes
    scale = args.scale
    batch_size = args.batch_size
    beta = 0.5      # dice在loss中的权重。loss = beta*dice + (1-beta)*cross_entropy + L2
    val_percent = 1
    amp = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path.parent.mkdir(exist_ok=True)

    # 随机数种子设置
    seed = 0
    torch.manual_seed(seed)    # 默认的随机数生成器，torch.Generator()可以新建生成器实例
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    assert args.name in unet.models
    logging.info(f'Loading model {args.model}')
    if args.name == "UNet":
        model = getattr(unet, args.name)(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    else:
        model = getattr(unet, args.name)(n_channels=args.channels, n_classes=args.classes, pretrained=False)
    model = model.to(memory_format=torch.channels_last)
    logging.info(f'Using device {device}')
    model.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 255])
    model.load_state_dict(state_dict)

    dataset = BasicDataset(dir_img, dir_mask, model.n_channels, scale, mask_values=mask_values)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val])
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    scores, pr_data_ave = evaluate(model, val_loader, device, amp, beta=beta)
    mean_score_dict = scores.calc_stat()
    logging.info(json.dumps(mean_score_dict, indent=4))
    pd.DataFrame(scores.get_score()).to_csv(save_path, index=True)

    from pylab import plt; plt.plot(pr_data_ave[:,1], pr_data_ave[:,0])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.savefig(str(save_path).replace('.csv', '_pr_curve.png'), dpi=500)
    np.save(str(save_path).replace('.csv', '_pr_curve.npy'), pr_data_ave)
