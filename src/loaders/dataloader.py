import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class LIDCDataset(Dataset):
    def __init__(self, labels_file, transform=None, train=True, auto_split=False, zero_indexed=True, chosen_chars=None, push=False):
        all_labels = pd.read_csv(labels_file)
        self.num_characteristics = len(all_labels.columns) - 1
        self.transform = transform
        self.zero_indexed = zero_indexed
        self.chosen_chars = chosen_chars

        if not auto_split:
            train_or_test_int = 1 if train else 0
            self.labels = all_labels[all_labels['train_or_test'] != train_or_test_int]
        else:
            train_df, test_df = train_test_split(all_labels, test_size=0.15, random_state=27)
            self.labels = train_df if train else test_df

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]) if transform and not push else None

        self.bweights = {}
        for char_index in range(1, self.num_characteristics + 1):  # 1-based index for characteristics
            if char_index == 2:  # Calcification uses different criteria
                pos_counts = (self.labels.iloc[:, char_index] == 6).sum()
                neg_counts = (self.labels.iloc[:, char_index] != 6).sum()
            else:
                pos_counts = (self.labels.iloc[:, char_index] > 3).sum()
                neg_counts = (self.labels.iloc[:, char_index] <= 3).sum()
            mean_counts = (pos_counts + neg_counts) / 2
            # mean_counts = max(mean_counts, 1)  # Avoid division by zero
            # pos_counts = max(pos_counts, 1)  # Avoid division by zero
            # neg_counts = max(neg_counts, 1)  # Avoid division by zero
            # self.bweights[char_index - 1] = [mean_counts / neg_counts, mean_counts / pos_counts]  # for binary classification
            self.bweights[char_index - 1] = torch.tensor([mean_counts / neg_counts, mean_counts / pos_counts], dtype=torch.float32)

        # Global final prediction weights
        pos_final_pred_counts = (self.labels.iloc[:, 1] > 3).sum()
        neg_final_pred_counts = (self.labels.iloc[:, 1] <= 3).sum()
        mean_final_pred_counts = (pos_final_pred_counts + neg_final_pred_counts) / 2
        self.bweight_final_pred = [mean_final_pred_counts / neg_final_pred_counts, mean_final_pred_counts / pos_final_pred_counts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.labels.iloc[index, 0]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        label_chars = []
        bweight_chars = []
        for char_index in range(2, self.num_characteristics + 1):
            if self.chosen_chars is not None and not self.chosen_chars[char_index - 1]:
                continue

            label_char = self.labels.iloc[index, char_index]
            if not self.zero_indexed:
                label_char -= 1

            if char_index == 2:
                binary_label_char = 1 if label_char == 6 else 0
            else:
                binary_label_char = 1 if label_char > 3 else 0

            # bweight_char = self.bweights[char_index - 1][binary_label_char] # for binary classification
            bweight_char = self.bweights[char_index - 1]
            label_chars.append(binary_label_char)
            bweight_chars.append(bweight_char)

        final_pred_label = 1 if (self.labels.iloc[index, 1] > 3) else 0
        bweight_fpred = self.bweight_final_pred[final_pred_label]

        if self.transforms:
            image = self.transforms(image)

        return image, label_chars, bweight_chars, final_pred_label, bweight_fpred
