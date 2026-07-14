import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from scipy.stats import norm


class LIDCDataset(Dataset):
    def __init__(self, labels_file, transform=None, chosen_chars=None, indeterminate=True, split='train', validation_split=0.10, test_split=0.10):
        all_labels = pd.read_csv(labels_file)
        self.transform = transform
        self.chosen_chars = chosen_chars

        # Preprocess the labels
        all_labels.drop(columns=['Internalstructure'], inplace=True)
        
        all_labels['Subtlety'] = all_labels['Subtlety'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        all_labels['Calcification'] = all_labels['Calcification'].replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1})
        all_labels['Sphericity'] = all_labels['Sphericity'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        all_labels['Margin'] = all_labels['Margin'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        all_labels['Lobulation'] = all_labels['Lobulation'].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1})
        all_labels['Spiculation'] = all_labels['Spiculation'].replace({1: 0, 2: 1, 3: 1, 4: 1, 5: 1})
        all_labels['Texture'] = all_labels['Texture'].replace({1: 0, 2: 0, 3: 0, 4: 0, 5: 1})
        all_labels['Diameter'] = all_labels['Diameter'].replace({1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
        
        if indeterminate:
            all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 1, 3: 1, 4: 2, 5: 2})
        else:
            # all_labels = all_labels[all_labels['Malignancy'] != 3]
            # all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 0, 4: 1, 5: 1})
            all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        
        # Extract patient identifiers from the image directory paths
        all_labels['patient_id'] = all_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))
        
        # Split the data into train, validation, and test sets
        random_state = 27
        unique_patients = all_labels['patient_id'].unique()    
        train_patients, temp_patients = train_test_split(unique_patients, test_size=(validation_split + test_split), random_state=random_state)
        val_patients, test_patients = train_test_split(temp_patients, test_size=(test_split / (validation_split + test_split)), random_state=random_state)
        
        # Assign the split data based on the chosen split
        if split == 'train':
            self.labels = all_labels[all_labels['patient_id'].isin(train_patients)]
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),  # Random rotation within 15 degrees
                transforms.RandomResizedCrop(100, scale=(0.8, 1.0), antialias=True),  # Random cropping and resizing back to original dimensions
            ])
        elif split == 'push':
            self.labels = all_labels[all_labels['patient_id'].isin(train_patients)]
            self.transforms = None
        elif split == 'val':
            self.labels = all_labels[all_labels['patient_id'].isin(val_patients)]
            self.transforms = None
        elif split == 'test':
            self.labels = all_labels[all_labels['patient_id'].isin(test_patients)]
            self.transforms = None
        else:
            raise ValueError("Invalid split name. Choose 'train', 'val', or 'test'.")

        # Extract the weights for each characteristic
        characteristics = self.labels[self.labels.columns[1:-2]]
        self.num_characteristics = len(characteristics.columns)
        malignancy = self.labels[['Malignancy']]
        
        class_counts = characteristics.apply(pd.Series.value_counts)
        self.char_weights = class_counts.apply(lambda x: (x.mean()) / x)

        class_counts = malignancy.apply(pd.Series.value_counts)
        self.malignancy_weights = class_counts.apply(lambda x: (x.mean()) / x)
        
        # Extract the number of slices of each nodule and the slice id
        self.labels['nodule_id'] = self.labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))+'-'+os.path.basename(os.path.dirname(x))))
        self.labels['slice_index'] = self.labels.groupby('nodule_id').cumcount()
        self.labels['total_slices'] = self.labels.groupby('nodule_id')['slice_index'].transform('max') + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the image
        path = self.labels['image_dir'].iloc[idx]
        array = np.load(path)
        image = torch.from_numpy(array)        
        
        # Get the weight for the slice
        num_slices = self.labels['total_slices'].iloc[idx]
        slice_index = self.labels['slice_index'].iloc[idx]
        x = np.linspace(0, num_slices-1, num_slices)
        mean = (num_slices - 1) / 2
        std_dev = 1.0
        weights = norm.pdf(x, mean, std_dev)
        weights = weights / weights.sum()
        weights = weights*10
        slice_weight = weights[slice_index]
        
        # Extract the labels and binary weights for each characteristic
        label_chars = []
        bweight_chars = []
        characteristics = self.labels[self.labels.columns[1:-4]]#-2]]
        for char_idx in range(0, self.num_characteristics):
            if self.chosen_chars[char_idx] is False:
                continue
            label = characteristics.iloc[idx, char_idx]
            label_chars.append(label)
            bweight_chars.append(self.char_weights.iloc[:, char_idx].values)
            
        # Extract the final prediction label and binary weight
        final_pred_label = self.labels['Malignancy'].iloc[idx]
        bweight_fpred = self.malignancy_weights.iloc[final_pred_label,0]
        # bweight_fpred = self.malignancy_weights.iloc[:,0].values #TODO: Uncomment this line and comment the above line for multi-class classification
        
        # Apply Data Augmentation to the image
        if self.transforms:
            image = self.transforms(image)
            
        return image, label_chars, bweight_chars, final_pred_label, bweight_fpred, slice_weight
