from os import path, listdir
import torch
from torchvision import transforms
import random

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_type: str, num_images=1000):
        if not path.exists(data_path):
            raise Exception(data_path + ' does not exist!')

        self.data = []

        images = listdir(data_path)

        current_data = [(path.join(data_path, image), data_type) for image in
                        images]
        self.data += current_data

        num_images = min(num_images, len(self.data))
        self.data = random.sample(self.data,
                                  num_images)  # only use num_images images

        # We use the transforms described in official PyTorch ResNet
        # inference example:
        # https://pytorch.org/hub/pytorch_vision_resnet/.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        try:
            image = Image.open(image_path)
        except:
            return None

        try:
            image = self.transform(
                image)  # some images in the dataset cannot be processed -
            # we'll skip them
        except Exception:
            return None

        dict_data = {
            'image':      image,
            'label':      label,
            'image_path': image_path
        }
        return dict_data


# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if
             sample]  # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)
