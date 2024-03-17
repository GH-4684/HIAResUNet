import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import datetime
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid
from pathlib import Path
from typing import Union
import json
import surface_distance as surfdist


Image.MAX_IMAGE_PIXELS = None

def get_rect(img: Image.Image, edge_color=255):
    img: np.ndarray = np.array(img.convert('L'))
    img = img!=edge_color
    sum_x = img.sum(axis=0)
    sum_y = img.sum(axis=1)
    left = (sum_x!=0).argmax()
    right = sum_x.size - 1 - (sum_x!=0)[::-1].argmax()
    top = (sum_y!=0).argmax()
    bottom = sum_y.size - 1 - (sum_y!=0)[::-1].argmax()
    return left, top, right, bottom


def plot_and_save_pred(img: Image.Image, mask_pred: np.ndarray, mask_true: np.ndarray, 
                      save_path: str, viz=False, methods=('raw','diff_rgb','contour','contour_mask','score'), extra_score: dict=None):
    """绘制和保存预测结果，计算和保存指标。classes=1或2。

    Args:
        img (Image.Image): 原图
        mask_pred (np.ndarray): 预测mask，0-1图像
        mask_true (np.ndarray): 真实mask，0-1图像
        save_path (str): 保存的文件路径，None则不保存
        viz (bool, optional): 是否显示绘图结果
        methods (tuple, optional): 结果输出方法。可选范围('raw','diff_rgb','diff_overlay','contour','contour_mask','score')
        extra_score (dict, optional): 保存到score中的附加指标，如运行时间
    """

    if save_path:
        save_path = Path(save_path)

    if 'score' in methods and save_path:
        path = save_path.with_name(save_path.stem+'_score.txt')
        score = Score(mask_pred, mask_true).get_score()
        if extra_score:
            score.update(extra_score)
        with open(path, 'w') as f:
            json.dump(score, f, indent=4)

    if mask_true is None:
        mask_true = np.zeros_like(mask_pred)
    mask_pred = mask_pred.astype(np.float64)
    mask_true = mask_true.astype(np.float64)
    H, W = mask_pred.shape
    assert mask_pred.max() <= 1 and mask_true.max() <= 1
    mask_pred = mask_pred * 255
    mask_true = mask_true * 255
    diff = mask_pred - mask_true

    colors = [(153,255,102), (240,240,240), (255,77,0), (102,102,204)]
    if 'raw' in methods:
        if save_path:
            path = save_path.with_name(save_path.stem+'_raw.png')
            Image.fromarray(mask_pred).convert('RGB').save(path)
        if viz:
            plt.figure(figsize=(6,6))
            plt.imshow(mask_pred)
            plt.axis('off')
    if 'diff_rgb' in methods:
        result_img = np.zeros((H, W, 3), dtype=np.uint8)
        if colors is not None:
            result_img[((diff==0)&(mask_pred>0))] = np.array(colors[0])     # TP
            result_img[((diff==0)&(mask_pred==0))] = np.array(colors[1])    # TN
            result_img[(diff>0)] = np.array(colors[2])                      # FP
            result_img[(diff<0)] = np.array(colors[3])                      # FN
        else:
            result_img[:, :, 0] = (diff > 0) * 255
            result_img[:, :, 1] = (diff == 0) * mask_true
            result_img[:, :, 2] = (diff < 0) * 255
        plt.figure(figsize=(6,6))
        plt.imshow(result_img)
        plt.axis('off')
        labels = ['TP', 'TN', 'FP', 'FN']
        for color,label in zip(colors,labels):
            plt.scatter([],[],edgecolors='black',linewidths=0.5,color=[c/255 for c in color],s=100,label=label)
        plt.legend(frameon=False)
        if save_path:
            path = save_path.with_name(save_path.stem+'_diff-rgb.png')
            plt.savefig(path, dpi=500, bbox_inches='tight')
            image = Image.open(path)
            image = image.crop(get_rect(image))  # (left, upper, right, lower)
            while True:
                try:
                    image.save(path)
                    break
                except:
                    pass
    if 'diff_overlay' in methods:
        result_img = np.zeros((H, W, 3), dtype=np.uint8)
        result_img[:, :, 0] = mask_pred
        result_img[:, :, 1] = mask_true
        if save_path:
            path = save_path.with_name(save_path.stem+'_diff-overlay.png')
            Image.fromarray(result_img).save(path)
        if viz:
            plt.figure(figsize=(6,6))
            plt.imshow(result_img)
            plt.axis('off')
    if 'contour' in methods:
        result_img = np.array(img.convert('RGB'), dtype=np.uint8)
        ret, binary = cv2.threshold(mask_pred, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (255,0,0), 2)
        if save_path:
            path = save_path.with_name(save_path.stem+'_contour.png')
            Image.fromarray(result_img).save(path)
        if viz:
            plt.figure(figsize=(6,6))
            plt.imshow(result_img)
            plt.axis('off')
    if 'contour_mask' in methods:
        img_arr = np.array(img.convert('RGB'), dtype=np.uint8)
        mask_pred_arr = np.tile(mask_pred[...,None], (1,1,3)) * 70 / 255
        result_img = (img_arr + mask_pred_arr).clip(0,255).astype(np.uint8)
        ret, binary = cv2.threshold(mask_pred, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (255,51,0), 2)
        if save_path:
            path = save_path.with_name(save_path.stem+'_contour-mask.png')
            Image.fromarray(result_img).save(path)
        if viz:
            plt.figure(figsize=(6,6))
            plt.imshow(result_img)
            plt.axis('off')

    if viz:
        plt.figure(figsize=(6,6))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        plt.close('all')


def save_feature_map(feature_map: Tensor, savepath: str, index=0):
    feature_map = feature_map[index].float()
    if feature_map.ndim == 2:
        feature_map = feature_map.unsqueeze(0)
    else:
        feature_map = feature_map.unsqueeze(1)
    nrow = np.power(2, np.ceil(np.log2(feature_map.shape[0])/2)).astype('i4')
    # nrow = np.ceil(np.sqrt(feature_map.shape[0])).astype('i4')
    image = make_grid(feature_map, nrow, padding=2, normalize=True, pad_value=1)
    image = image.cpu().detach().numpy().transpose((1, 2, 0))
    plt.imsave(savepath, image)


def plot_tensors(t_dict: dict, nrow: int, save=True):
    num = len(t_dict.items())
    ncol = int(np.ceil(num/nrow))
    for i,(k,v) in enumerate(t_dict.items()):
        plt.subplot(nrow, ncol, i+1)
        plt.title(k)
        if v is None: continue
        plt.imshow(v.cpu().detach().numpy().mean(axis=(0,1)), cmap='gray')
        plt.axis(False)
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        os.makedirs('output_vis_tensors', exist_ok=True)
        t = datetime.datetime.now().strftime('%H-%M-%S-%f')
        plt.savefig(f'output_vis_tensors/{t}.png', dpi=500)


def plot_save_loss(data, savedir: Path = None, save_data = True, save_fig = True):
    # data: (epoch, step, train_loss, val_loss, test_loss, val_dice, test_dice)
    data = np.array(data, dtype=np.float32)
    epoch = np.nanmax(data[:,0])

    if save_data and savedir is not None:
        np.save(savedir/f'data_epoch{epoch:02.0f}.npy', data)

    if save_fig and savedir is not None:
        config = {
            "font.family":'serif',
            "font.size": 16.8,
            "mathtext.fontset":'stix',
            "font.serif": ['times new roman'],
        }
        plt.rcParams.update(config)

        plt.figure(figsize=(8,6))
        for n in range(2):
            index = ~np.isnan(data[:,n+2])
            x = np.linspace(0, epoch, index.sum())
            y = data[index,n+2]
            plt.plot(x,y)
        plt.ylim([0, np.nanmax(data[:,2])*1.1])
        plt.xlabel('eopchs')
        plt.ylabel('loss')
        plt.title('loss curve')

        plt.twinx()
        for n in range(1):
            index = ~np.isnan(data[:,n+5])
            x = np.linspace(0, epoch, index.sum())
            y = data[index,n+5]
            plt.plot(x,y,'C2')
        plt.ylim([0, 1])
        plt.ylabel('Dice score')
        plt.gcf().legend(('train loss', 'val loss', 'val dice'), bbox_to_anchor=(1,0.86), bbox_transform=plt.gca().transAxes)

        plt.savefig(savedir/'loss curve.png', dpi=500, bbox_inches='tight')


class Score:
    def __init__(self, pred_mask: np.ndarray, true_mask: np.ndarray=None, classes: int=1) -> None:
        if true_mask is None:
            self.score = dict(
                P=None,
                R=None,
                ACC=None,
                F1=None,
                IoU=None,
                Dice=None,
                Kappa=None,
                ASSD=None,
                HD95=None,
                HD100=None
            )
            return

        assert pred_mask.shape == true_mask.shape
        if pred_mask.ndim == 2 and true_mask.ndim == 2 and classes == 1:    # [H,W]
            pred_mask = np.stack((pred_mask, 1-pred_mask), axis=2)
            true_mask = np.stack((true_mask, 1-true_mask), axis=2)
        else:
            raise Exception()
        self.H, self.W, N = true_mask.shape
        cm = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                cm[i, j] = (true_mask[:,:,i] * pred_mask[:,:,j]).sum()
        self.cm = cm
        self.num = self.H * self.W

        self.surface_distances = surfdist.compute_surface_distances(
                pred_mask[:,:,0].astype(np.bool_), true_mask[:,:,0].astype(np.bool_), spacing_mm=(1.0, 1.0))

        self.pred_mask = pred_mask
        self.true_mask = true_mask
        self.classes = classes
        
        self.score = dict(
            P=self.precision,
            R=self.recall,
            ACC=self.accuracy,
            F1=self.F1,
            IoU=self.IoU,
            Dice=self.dice,
            Kappa=self.kappa,
            ASSD=self.assd,
            HD95=self.hd_95,
            HD100=self.hd_100
        )

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

    def get_score(self) -> dict:
        return self.score
