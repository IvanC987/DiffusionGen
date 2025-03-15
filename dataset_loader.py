import os
import random
import torch
from PIL import Image


class DatasetLoader:
    def __init__(self, img_dir: str, prompt_path: str, batch_size: int, device: str, train_split: float):
        """
        A simple dataset loader

        :param img_dir: Path to the image directory
        :param prompt_path: Path to the prompt txt file, should be line-separated
        :param batch_size: Batch size
        :param device: Device to use (e.g. cpu or cuda)
        :param train_split: A float, ratio of train to val loss. Usually 0.9 or 0.95
        """

        assert 0.5 <= train_split < 1.0, f"Value of {train_split=} is not allowed"

        self.batch_size = batch_size
        self.device = device

        prompts = self.load_prompts(prompt_path)
        image_paths = [os.path.join(img_dir, p) for p in os.listdir(img_dir)]

        # Sorting based on my dataset filename structure (img_1.png, img_2.png, ...)
        image_paths = sorted(image_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        assert len(image_paths) == len(prompts), f"Mismatch between {len(image_paths)=} and {len(prompts)=}"


        prompt_img_pairs = [(p, i) for p, i in zip(prompts, image_paths)]

        random.shuffle(prompt_img_pairs)

        n = int(len(prompt_img_pairs) * train_split)
        self.train_pairs = prompt_img_pairs[:n]
        self.val_pairs = prompt_img_pairs[n:]

        self.train_idx = 0
        self.val_idx = 0

        self.train_epoch = 0
        self.val_epoch = 0

        print("\n****************"
              "\nImportant Note- MAKE SURE the prompt-image pairs are correctly aligned!"
              "\nHere are the first few pairs for verification: "
              f"\n{self.train_pairs[:5]=}"
              "\n****************\n")


    @staticmethod
    def load_prompts(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = f.read().split("\n")
            prompts = [p for p in prompts if len(p) > 0]

        return prompts


    def get_batch(self, train: bool):
        """
        Returns a tensor of shape (b, 3, h, w) normalized to range [-1, 1].
        Assumes all images within given dir is of same shape

        :param train: Boolean of whether to return data from train or validation split
        :return: A tuple (prompt_batch, images) where prompt_batch is a list of strings, images is a tensor of shape (b, 3, h, w)
        """

        if train:
            batch = self.train_pairs[self.train_idx: self.train_idx + self.batch_size]
            self.train_idx += self.batch_size

            if self.train_idx + self.batch_size >= len(self.train_pairs):
                self.train_idx = 0
                self.train_epoch += 1
                random.shuffle(self.train_pairs)
        else:
            batch = self.val_pairs[self.val_idx: self.val_idx + self.batch_size]
            self.val_idx += self.batch_size

            if self.val_idx + self.batch_size >= len(self.val_pairs):
                self.val_idx = 0
                self.val_epoch += 1
                random.shuffle(self.val_pairs)


        prompt_batch = [e[0] for e in batch]
        img_batch = [e[1] for e in batch]

        # Convert to PIL Images
        images = [Image.open(i).convert("RGB") for i in img_batch]

        # Get their dimensions
        width, height = images[0].size

        # Convert into numerical repr
        images = torch.tensor([obj.getdata() for obj in images], dtype=torch.float32, device=self.device)

        images = images.reshape(len(images), width, height, 3)  # Reshape from (b, w*h, 3) -> (b, w, h, 3)
        images = images.permute(0, 3, 2, 1)  # Permute into (b, 3, h, w)
        # Normalize the tensor to range [-1, 1]
        images = ((images / 255) - 0.5) * 2

        return prompt_batch, images



