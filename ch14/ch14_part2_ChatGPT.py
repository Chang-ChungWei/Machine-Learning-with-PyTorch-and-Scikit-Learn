# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:42:21 2024

@author: user
"""

import os
import requests
import zipfile
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision 
from torchvision import transforms 
from torch.utils.data import DataLoader, Subset

# Function to download files from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk:
                f.write(chunk)


def download_and_extract(url, download_path, extract_to):
    if not os.path.exists(download_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Download complete.")

    if not os.path.exists(extract_to):
        print("Extracting files...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")


# URLs for the dataset and annotation files
img_zip_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
identity_txt_url = "https://drive.google.com/uc?export=download&id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS"
attr_txt_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
bbox_txt_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pbThiMVRxWXZ4dU0"
landmarks_align_txt_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pd0FJY3Blby1HUTQ"
landmarks_txt_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pTzJIdlJWdHczRlU"
eval_partition_txt_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NSMzRuSXJEVkk"

# File paths for the dataset and annotation files
image_zip_path = "./img_align_celeba.zip"
image_extract_path = "./celeba"
identity_txt_path = os.path.join(image_extract_path, "identity_CelebA.txt")
attr_txt_path = os.path.join(image_extract_path, "list_attr_celeba.txt")
bbox_txt_path = os.path.join(image_extract_path, "list_bbox_celeba.txt")
landmarks_align_txt_path = os.path.join(image_extract_path, "list_landmarks_align_celeba.txt")
landmarks_txt_path = os.path.join(image_extract_path, "list_landmarks_celeba.txt")
eval_partition_txt_path = os.path.join(image_extract_path, "list_eval_partition.txt")

# Create directory if it doesn't exist
os.makedirs(image_extract_path, exist_ok=True)

# Download and extract the dataset and annotation files
download_and_extract(img_zip_url, image_zip_path, image_extract_path)
download_file_from_google_drive("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", identity_txt_path)
download_file_from_google_drive("0B7EVK8r0v71pblRyaVFSWGxPY0U", attr_txt_path)
download_file_from_google_drive("0B7EVK8r0v71pbThiMVRxWXZ4dU0", bbox_txt_path)
download_file_from_google_drive("0B7EVK8r0v71pd0FJY3Blby1HUTQ", landmarks_align_txt_path)
download_file_from_google_drive("0B7EVK8r0v71pTzJIdlJWdHczRlU", landmarks_txt_path)
download_file_from_google_drive("0B7EVK8r0v71pY0NSMzRuSXJEVkk", eval_partition_txt_path)

# Load the dataset using torchvision
celeba_train_dataset = torchvision.datasets.CelebA(image_extract_path, split='train', target_type='attr', download=True)
celeba_valid_dataset = torchvision.datasets.CelebA(image_extract_path, split='valid', target_type='attr', download=True)
celeba_test_dataset = torchvision.datasets.CelebA(image_extract_path, split='test', target_type='attr', download=True)

print('Train set:', len(celeba_train_dataset))
print('Validation set:', len(celeba_valid_dataset))
print('Test set:', len(celeba_test_dataset))

# Example of image transformation and data augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

get_smile = lambda attr: attr[18]

celeba_train_dataset = torchvision.datasets.CelebA(image_extract_path, 
                                                   split='train', 
                                                   target_type='attr', 
                                                   download=False, 
                                                   transform=transform_train,
                                                   target_transform=get_smile)

celeba_valid_dataset = torchvision.datasets.CelebA(image_extract_path, 
                                                   split='valid', 
                                                   target_type='attr', 
                                                   download=False, 
                                                   transform=transform,
                                                   target_transform=get_smile)

celeba_test_dataset = torchvision.datasets.CelebA(image_extract_path, 
                                                   split='test', 
                                                   target_type='attr', 
                                                   download=False, 
                                                   transform=transform,
                                                   target_transform=get_smile)

celeba_train_dataset = Subset(celeba_train_dataset, torch.arange(16000)) 
celeba_valid_dataset = Subset(celeba_valid_dataset, torch.arange(1000)) 
 
print('Train set:', len(celeba_train_dataset))
print('Validation set:', len(celeba_valid_dataset))

batch_size = 32

torch.manual_seed(1)
train_dl = DataLoader(celeba_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(celeba_valid_dataset, batch_size, shuffle=False)
test_dl = DataLoader(celeba_test_dataset, batch_size, shuffle=False)

# Example of visualizing the dataset
fig = plt.figure(figsize=(16, 8.5))

# Column 1: cropping to a bounding-box
ax = fig.add_subplot(2, 5, 1)
img, attr = celeba_train_dataset[0]
ax.set_title('Crop to a \nbounding-box', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 6)
img_cropped = transforms.functional.crop(img, 50, 20, 128, 128)
ax.imshow(img_cropped)

# Column 2: flipping (horizontally)
ax = fig.add_subplot(2, 5, 2)
img, attr = celeba_train_dataset[1]
ax.set_title('Flip (horizontal)', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 7)
img_flipped = transforms.functional.hflip(img)
ax.imshow(img_flipped)

# Column 3: adjust contrast
ax = fig.add_subplot(2, 5, 3)
img, attr = celeba_train_dataset[2]
ax.set_title('Adjust constrast', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast = transforms.functional.adjust_contrast(img, contrast_factor=2)
ax.imshow(img_adj_contrast)

# Column 4: adjust brightness
ax = fig.add_subplot(2, 5, 4)
img, attr = celeba_train_dataset[3]
ax.set_title('Adjust brightness', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness = transforms.functional.adjust_brightness(img, brightness_factor=1.3)
ax.imshow(img_adj_brightness)

# Column 5: cropping from image center 
ax = fig.add_subplot(2, 5, 5)
img, attr = celeba_train_dataset[4]
ax.set_title('Center crop\nand resize', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 10)
img_center_crop = transforms.functional.center_crop(img, [0.7*218, 0.7*178])
img_center_crop = transforms.functional.resize(img_center_crop, [218, 178])
ax.imshow(img_center_crop)

fig.tight_layout()
