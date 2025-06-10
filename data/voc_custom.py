"""Custom VOC Dataset compatible with VOC format but without year directories."""
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from .voc0712 import VOCAnnotationTransform, VOC_CLASSES

class VOCCustomDetection(data.Dataset):
    """VOC Detection Dataset without year-specific subfolders."""
    def __init__(self, root, image_sets=['trainval'],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOCCUSTOM'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for name in image_sets:
            for line in open(osp.join(self.root, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
