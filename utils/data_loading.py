import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dirs: list = None, mask_dirs: list = None, channels: int = None, 
                 scale: float = 1.0, mask_values: list = None, mask_suffix: str = '_mask'):
        self.trans = transforms.Compose([
                        # transforms.ToPILImage(),
                        # MyColorJitter(brightness=0.15, contrast=0.25, saturation=0.25, k=0.35), # 1-x, 1+x
                        transforms.ToTensor(),
                    ])

        if images_dirs is not None:
            self.images_dirs = [Path(images_dir) for images_dir in images_dirs]
            self.mask_dirs = [Path(mask_dir) for mask_dir in mask_dirs]
            self.channels = channels
        else:
            return
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = []
        self.ids_image_abspath = []
        self.ids_mask_abspath = []
        for i, images_dir in enumerate(self.images_dirs):
            for file in listdir(images_dir):
                if isfile(join(images_dir, file)) and not file.startswith('.'):
                    name, ext = splitext(file)
                    assert ext in ['.png', '.jpg']
                    self.ids.append(name)
                    img_file: Path = list(self.images_dirs[i].glob(name + '.*'))
                    assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
                    self.ids_image_abspath.append(img_file[0])
                    mask_file: Path = list(self.mask_dirs[i].glob(name + self.mask_suffix + '.*'))
                    assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
                    self.ids_mask_abspath.append(mask_file[0])

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dirs}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        # unique = []
        # for i in tqdm(range(len(self.ids))):
        #     unique.append(unique_mask_values(self.ids[i], self.mask_dir, self.mask_suffix))

        if mask_values is None:
            unique = []
            with Pool() as p:
                for mask_dir in self.mask_dirs:
                    unique += list(tqdm(
                        p.imap(partial(unique_mask_values, mask_dir=mask_dir, mask_suffix=self.mask_suffix), self.ids),
                        total=len(self.ids)
                    ))
            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        else:
            if type(mask_values[0]) != int:
                mask_values = [v.item() for v in mask_values]
            self.mask_values = sorted(set(mask_values))

        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, mask_values, pil_img, channels, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.full((newH, newW), -1, dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    img = img[:, :, :3]
                    mask[(img == v).all(-1)] = i

            assert np.sum(mask<0) == 0, 'mask_values error'
            return torch.as_tensor(mask.copy()).long().contiguous()

        else:
            if img.ndim == 2:
                img = img[..., np.newaxis]
                img = np.repeat(img, 3, axis=2)
            
            img = self.trans(img)

            img = img[:channels, :, :]
            
            return img.float().contiguous()

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # img_file = list(self.images_dir.glob(name + '.*'))

        # assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # mask = load_image(mask_file[0])
        # img = load_image(img_file[0])

        mask = load_image(self.ids_mask_abspath[idx])
        img = load_image(self.ids_image_abspath[idx])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.channels, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.channels, self.scale, is_mask=True)

        return {
            'image': img,
            'mask': mask,
            'name': str(self.ids[idx]),
            'image_path': str(self.ids_image_abspath[idx]),
            'mask_path': str(self.ids_mask_abspath[idx]),
        }


# class CarvanaDataset(BasicDataset):
#     def __init__(self, images_dir, mask_dir, scale=1):
#         super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class MyColorJitter(object):
    def __init__(self, brightness=0.15, contrast=0.25, saturation=0.25, k=0.35):
        self.colorJitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=0.0)
        self.k = k

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        img = self.colorJitter(img_pil)
        img = np.array(img)
        assert img.dtype == np.uint8 and img.shape[2] == 3
        img = (img / 255 + np.random.random((1,1,3)) * self.k) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img
