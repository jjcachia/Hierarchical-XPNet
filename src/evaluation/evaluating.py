import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import norm

class LIDCEvaluationDataset(Dataset):
    def __init__(self, labels_file, transform=None, chosen_chars=None, indeterminate=True, validation_split=0.10, test_split=0.10):
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
            all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})
        else:
            all_labels['Malignancy'] = all_labels['Malignancy'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
        
        # Extract patient identifiers from the image directory paths
        all_labels['patient_id'] = all_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))
        
        # Get the test labels
        random_state = 27
        unique_patients = all_labels['patient_id'].unique()    
        _, temp_patients = train_test_split(unique_patients, test_size=(validation_split + test_split), random_state=random_state)
        _, test_patients = train_test_split(temp_patients, test_size=(test_split / (validation_split + test_split)), random_state=random_state)
        test_labels = all_labels[all_labels['patient_id'].isin(test_patients)]
        
        # Group the test labels by nodule
        # test_labels['nodule_id'] = test_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))+'-'+os.path.basename(os.path.dirname(x))))
        test_labels.loc[:, 'nodule_id'] = test_labels['image_dir'].apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))+'-'+os.path.basename(os.path.dirname(x))))

        self.nodule_labels = test_labels.groupby('nodule_id')
        self.nodule_keys = list(self.nodule_labels.groups.keys())

    def __len__(self):
        return len(self.nodule_labels)

    def __getitem__(self, idx):
        nodule_key = self.nodule_keys[idx]
        nodule_data = self.nodule_labels.get_group(nodule_key)
        
        # Load all slices for the nodule and the label of the nodule
        images = [np.load(row['image_dir']) for _, row in nodule_data.iterrows()]
        # images = [np.expand_dims(img, axis=0) for img in images]
        # images = [np.repeat(img, 3, axis=0) for img in images]
        images = [torch.from_numpy(img) for img in images]

        label_chars = []
        bweight_chars = []
        characteristics = nodule_data[nodule_data.columns[1:-3]]
        for char_idx in range(0, len(characteristics.columns)):
            if self.chosen_chars[char_idx] is False:
                continue
            label = characteristics.iloc[0, char_idx]
            label_chars.append(label)        
        
        # Convert images if a transform is specified
        # if self.transform:
        #     images = [self.transform(Image.fromarray(img)) for img in images]
        
        final_pred_label = nodule_data.iloc[0]['Malignancy']  # Assuming 'Malignancy' is the last label

        return torch.stack(images), label_chars, final_pred_label
    