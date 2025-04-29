import os.path
import queue

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensor

from config import PROJECT_DIR, device, models_path
from face_aligner import align_image
from face_aligner import get_detector
from torchvision import transforms
from pix2pix_training import UNetGenerator
import torch.nn as nn
from PIL import Image


def get_image_path(image_name):
    return os.path.join(PROJECT_DIR, 'photos', image_name)


def load_generator(ckpt_path: str, device="cuda") -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    netG = UNetGenerator().to(device)
    state_dict = ckpt.get("G", ckpt)
    netG.load_state_dict(state_dict, strict=False)
    netG.eval()
    return netG
    

class Worker:
    def __init__(self):
        self.landmarks_detector = get_detector()
        self.model = {'merged': load_generator(models_path['merged'], device),
                      # 'cartoon': torch.jit.load(models_path['cartoon'], map_location=device)
                     }
        self.transforms = transforms.Compose([
            transforms.Resize((1024, 1024), Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])

        print('Init was succesfully completed')

    def crop_faces(self, image_name: str):
        image_path = get_image_path(image_name)
        faces = list(align_image(self.landmarks_detector, image_path))
        return faces

    def load_image(self, path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def predict(self, image: np.array, mode: str):
        print(image.shape)
        # image = np.moveaxis(image, -1, 0)
        # print(image.shape)
        image = self.transforms(Image.fromarray(image)).unsqueeze(0).to(device)
        gen = self.model[mode](image)
        gen = (gen.squeeze(0).cpu() + 1) / 2  # [0,1]
        # to uint8
        gen = (gen.permute(1,2,0).detach().numpy()*255).clip(0,255).astype(np.uint8)
        gen = gen.astype(np.uint8)
        return gen
