import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

class LIDCDataset(Dataset):
    def __init__(self, labels_file, transform=None, train=True, num_classes=3, 
                 auto_split=False, zero_indexed=True, chosen_chars=None, push=False):
        """
        Initializes the LIDCDataset.

        Args:
            labels_file (str): The path to the CSV file containing the labels.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
            train (bool, optional): Specifies whether to load the training data or the test data. Default is True.
            num_classes (int, optional): The number of classes in the dataset. Default is 3.
            auto_split (bool, optional): Specifies whether to split the data into train and test sets automatically. Default is False.
            zero_indexed (bool, optional): Specifies whether the labels are zero-indexed. Default is True.
            chosen_chars (list, optional): A list of chosen characteristics. Default is None.
            push (bool, optional): Specifies whether to use the push dataset. Default is False.
        """
        all_labels = pd.read_csv(labels_file)

        # Determine the number of characteristics by the number of columns in the CSV file
        self.num_characteristics = len(all_labels.columns) - 1

        # Depending on whether the auto_split flag is set, the data is either split based on a predefined train/test column 
        # or randomly split into training and testing sets using a 75%/25% ratio
        train_or_test_int = 1 if train else 0
        self.transforms = None
        if not auto_split:
            self.labels = all_labels[all_labels.train_or_test != train_or_test_int] # Split based on train or test column
        else:
            train_df, test_df = train_test_split(all_labels, test_size=0.15, random_state=27)   # Split randomly into training and testing sets
            train_push_df = train_df
            if train: # if train, then use the training data
                if not push: # if not push, transform the data using random horizontal and vertical flips
                    self.labels = train_df
                    self.transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip()
                    ])
                else: # if push, use the training data as is
                    self.labels = train_push_df
            else:
                self.labels = test_df

        self.transform = transform
        self.num_classes = num_classes
        self.zero_indexed = zero_indexed
        self.chosen_chars = chosen_chars

        
        '''The class also calculates class weights for balancing the dataset. It does this by counting the occurrences of each class 
        for each characteristic and then computing a weight as the mean count divided by the individual class count. 
        Additionally, it calculates weights for the final predictions based on whether the scores are greater than 
        or less than or equal to 3, aiming to balance positive and negative predictions.'''
        features = all_labels.columns[1:]
        class_counts = all_labels[features].apply(pd.Series.value_counts) # class_counts contains the count of each class for each characteristic
        self.bweights = class_counts.apply(lambda x: (x.mean()) / x) # bweight = (mean of all classes) / (class count)

        pos_final_pred_counts = (self.labels.iloc[:, 1] > 3).sum() # final prediction is positive if the score is greater than 3
        neg_final_pred_counts = (self.labels.iloc[:, 1] <= 3).sum() # final prediction is negative if the score is less than or equal to 3
        mean_final_pred_counts = (pos_final_pred_counts + neg_final_pred_counts) / 2 # mean of the final predictions
        self.bweight_final_pred = [mean_final_pred_counts / neg_final_pred_counts, mean_final_pred_counts / pos_final_pred_counts] # bweight = (mean of all classes) / (class count)


    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        dataset_length = len(self.labels)
        return dataset_length

    def __getitem__(self, index):
        """
        Retrieves the item at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the following elements:
                - image (PIL.Image.Image): The image at the specified index.
                - label_chars (list): A list of one-hot encoded label characters.
                - bweight_chars (list): A list of bweight characters.
                - final_pred_label (int): The final predicted label.
                - bweight_fpred (float): The bweight for the final predicted label.
        """
        path = self.labels.iloc[index, 0]

        image = Image.open(path)
        if self.transform:
            image = self.transform(image)

        label_chars = []
        bweight_chars = []
        for char_index in range(2, self.num_characteristics+1):
            if self.chosen_chars is not None:
                if not(self.chosen_chars[char_index-1]):
                    continue

            label_char = self.labels.iloc[index, char_index]

            if not(self.zero_indexed):
                label_char -= 1
                # print("WARNING: Not zero indexed")

            bweight_char = self.bweights.iloc[label_char, char_index-1] #char_index-1 because bweights only contains characteristics columns (and no directories column)

            label_char_one_hot = np.eye(self.num_classes)[label_char]

            label_chars.append(label_char_one_hot)
            bweight_chars.append(bweight_char)

        final_pred_label = 1 if (self.labels.iloc[index, 1] > 3) else 0

        bweight_fpred = self.bweight_final_pred[final_pred_label]

        if self.transforms:
            image = self.transforms(image)

        return image, label_chars, bweight_chars, final_pred_label, bweight_fpred
        
