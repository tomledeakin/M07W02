import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET


class MyDataset(Dataset):
    def __init__(self, annotations_dir, image_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = self.filter_images_with_multiple_objects()

    def filter_images_with_multiple_objects(self):
        valid_image_files = []
        for f in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir, f)):
                img_name = f
                annotation_name = os.path.splitext(img_name)[0] + ".xml"
                annotation_path = os.path.join(self.annotations_dir, annotation_name)

                if self.count_objects_in_annotation(annotation_path) == 1:
                    valid_image_files.append(img_name)
        return valid_image_files

    def count_objects_in_annotation(self, annotation_path):
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            count = 0
            for obj in root.findall("object"):
                count += 1
            return count
        except FileNotFoundError:
            return 0

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Get image size for normalization
        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        label = None
        bbox = None
        for obj in root.findall("object"):
            name = obj.find("name").text
            if label is None:  # Take the first label
                label = name
            # Get bounding box coordinates
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # Normalize bbox coordinates to [0, 1]
            bbox = [
                xmin / image_width,
                ymin / image_height,
                xmax / image_width,
                ymax / image_height,
            ]

        # Convert label to numerical representation (0 for cat, 1 for dog)
        label_num = 0 if label == "cat" else 1 if label == "dog" else -1

        return label_num, torch.tensor(bbox, dtype=torch.float32)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img1_file = self.image_files[idx]
        img1_path = os.path.join(self.image_dir, img1_file)

        annotation_name = os.path.splitext(img1_file)[0] + ".xml"
        img1_annotations = self.parse_annotation(os.path.join(self.annotations_dir, annotation_name))

        idx2 = random.randint(0, len(self.image_files) - 1)
        img2_file = self.image_files[idx2]
        img2_path = os.path.join(self.image_dir, img2_file)

        annotation_name = os.path.splitext(img2_file)[0] + ".xml"
        img2_annotations = self.parse_annotation(os.path.join(self.annotations_dir, annotation_name))

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Horizontal merge
        merged_image = Image.new(
            "RGB", (img1.width + img2.width, max(img1.height, img2.height))
        )
        merged_image.paste(img1, (0, 0))
        merged_image.paste(img2, (img1.width, 0))

        merged_w = img1.width + img2.width
        merged_h = max(img1.height, img2.height)

        merged_annotations = []

        # No change for objects from img1
        merged_annotations.append({"bbox": img1_annotations[1], "label": img1_annotations[0]})
        # Adjust bbox coordinates for objects from img2
        new_bbox = [
            img2_annotations[1][0] + img1.width,
            img2_annotations[1][1],
            img2_annotations[1][2] + img1.width,
            img2_annotations[1][3],
        ]
        merged_annotations.append({"bbox": new_bbox, "label": img2_annotations[0]})

        # Split the merged image into 4 patches
        patch_width = merged_w // 2
        patch_height = merged_h // 2
        patches = []
        patch_annotations = []  # list of dictionary

        for i in range(2):
            for j in range(2):
                left = j * patch_width
                upper = i * patch_height
                right = (j + 1) * patch_width
                lower = (i + 1) * patch_height

                patch = merged_image.crop((left, upper, right, lower))
                patches.append(patch)

                current_patch_annotations = []
                for anno in merged_annotations:
                    center_x, center_y = self.calculate_center(anno["bbox"])

                    # Check if the center of the ground truth bbox is within the patch
                    if left <= center_x < right and upper <= center_y < lower:
                        current_patch_annotations.append({"bbox": anno["bbox"], "label": anno["label"]})
                        break

                if len(current_patch_annotations) == 0:
                    current_patch_annotations.append({"bbox": [0, 0, 0, 0], "label": 0})  # dummy bbox

                patch_annotations.append(current_patch_annotations[0])

        # Transform and convert to tensors
        if self.transform:
            patches = [self.transform(patch) for patch in patches]

        patches = torch.stack(patches)  # shape (4, C, H, W)
        patch_annotations = [
            {
                "bbox": torch.tensor(patch["bbox"], dtype=torch.float32),
                "label": torch.tensor(patch["label"], dtype=torch.long),
            }
            for patch in patch_annotations
        ]

        return patches, patch_annotations

    @staticmethod
    def calculate_center(bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
