import os
from PIL import Image

from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, json_file, root_dir, transforms=None):
        self.coco = COCO(json_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        image_id = self.ids[index]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        image_info = self.coco.loadImgs(image_id)[0]
        image = Image.open(os.path.join(self.root_dir, image_info['file_name'])).convert("RGB")

        boxes = []
        labels = []
        for ann in annotations:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)