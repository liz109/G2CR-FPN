import numpy as np
import pandas as pd
import torch.utils.data as Data
from torchvision import transforms
import cv2

# for Grayscale
transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.5], std=[0.5]) 
])  

class AAPMDataset(Data.Dataset):
    def __init__(self, annotation, resize=False, transform=transform, patch_n=None, patch_size=None) -> None:
        super().__init__()
        self.df = pd.read_pickle(annotation)
        self.transform = transform
        self.resize = resize
        self.patch_n = patch_n
        self.patch_size = patch_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        input_img, target_img, file = info['input'], info['target'], info['file']
        input_img, target_img = np.load(input_img), np.load(target_img)

        if self.resize:
            dim = (input_img.shape[1]//2, input_img.shape[0]//2)
            input_img = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)
            target_img = cv2.resize(target_img, dim, interpolation=cv2.INTER_AREA)


        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = self.get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            # patches: size(1, patch_n=10, patch_size=64, patch_size=64)
            return (input_patches, target_patches, file)
        else:
            # img: size(1, 512, 512), size(1, 512, 512)
            return (input_img, target_img, file)
        

    # random select [patch_n] patches with size [patch_size, patch_size]
    def get_patch(self, full_input_img, full_target_img, patch_n, patch_size):
        assert full_input_img.shape == full_target_img.shape
        patch_input_imgs = []
        patch_target_imgs = []
        _, h, w = full_input_img.shape  # size(1, 512, 512)
        new_h, new_w = patch_size, patch_size
        for _ in range(patch_n):
            top = np.random.randint(0, h-new_h)
            left = np.random.randint(0, w-new_w)
            patch_input_img = full_input_img[:, top:top+new_h, left:left+new_w]
            patch_target_img = full_target_img[:, top:top+new_h, left:left+new_w]
            patch_input_imgs.append(patch_input_img)
            patch_target_imgs.append(patch_target_img)
        return np.array(patch_input_imgs), np.array(patch_target_imgs)
            
        

        