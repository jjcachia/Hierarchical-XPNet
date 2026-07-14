import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from tqdm import tqdm

from src.utils.helpers import makedir, save_pickle

def push_prototypes(
    dataloader,  # pytorch dataloader
    model,  # pytorch network with feature encoder and prototype vectors
    device,
    class_specific=True,  # enable pushing protos from only the alotted class
    abstain_class=False,  # indicates K+1-th class is of the "abstain" type
    preprocess_input_function=None,  # normalize if needed
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved in this dir
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    replace_prototypes=True
):
    """
    Search the training set for image patches that are semantically closest to
    each learned prototype, then updates the prototypes to those image patches.

    To do this, it computes the image patch embeddings (IPBs) and saves those
    closest to the prototypes. It also saves the prototype-to-IPB distances and
    predicted occurrence maps.

    If abstain_class==True, it assumes num_classes actually equals to K+1, where
    K is the number of real classes and 1 is the extra "abstain" class for
    uncertainty estimation.
    """

    model.eval()
    print(f"############## push at epoch {epoch_number} #################")

    # creating the folder (with epoch number) to save the prototypes' info and visualizations
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, "epoch-" + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    # find the number of prototypes, and number of classes for this push
    # prototype_shape = (model.prototypes_per_characteristic, model.prototype_shape[1], model.prototype_shape[2], model.prototype_shape[3])  # shape (P, D, 1, 1)
    P = model.prototypes_per_characteristic
    num_characteristics = model.num_characteristics
    num_classes = model.num_classes
    
    proto_class_specific = np.full(P, class_specific)
    
    if abstain_class:
        K = num_classes - 1
        assert K >= 2, "Abstention-push must have >= 2 classes not including abstain"
        # for the uncertainty prototypes, class_specific is False
        # for now assume that each class (inc. unc.) has P_per_class == P/num_classes
        P_per_class = P // num_classes
        proto_class_specific[K * P_per_class : P] = False
    else:
        K = num_classes

    # keep track of the input embedding closest to each prototype
    proto_dist_ = [np.full(P, np.inf) for _ in range(num_characteristics)]  # saves the distances to prototypes (distance = 1-CosineSimilarities). shape (P)
    # save some information dynamically for each prototype
    # which are updated whenever a closer match to prototype is found
    occurrence_map_ = [[None for _ in range(P)] for _ in range(num_characteristics)] # saves the computed occurence maps. shape (P, 1, H, W)
    # saves the input to prototypical layer (conv feature * occurrence map), shape (P, D)
    protoL_input_ = [[None for _ in range(P)] for _ in range(num_characteristics)]
    # saves the input images with embeddings closest to each prototype. shape (P, 3, Ho, Wo)
    image_ = [[None for _ in range(P)] for _ in range(num_characteristics)]
    # saves the gt label. shape (P)
    gt_ = [[None for _ in range(P)] for _ in range(num_characteristics)]
    # saves the prediction logits of cases seen. shape (P, K)
    pred_ = [[None for _ in range(P)] for _ in range(num_characteristics)]
    # saves the filenames of cases closest to each prototype. shape (P)
    filename_ = [[None for _ in range(P)] for _ in range(num_characteristics)] # TODO: add filename in getitem of dataloader

    # data_iter = iter(dataloader)
    # iterator = tqdm(range(len(dataloader)), dynamic_ncols=True)
    for X, y, _, _, _, _, path in tqdm(dataloader, leave=False):
        # data_sample = next(data_iter)
        # x = data_sample["cine"]  
        
        if preprocess_input_function is not None:
            X = preprocess_input_function(X)

        # get the network outputs for this instance
        with torch.no_grad():
            x = X.to(device)    # shape (B, 3, Ho, Wo)
            (
                protoL_input_torch,
                proto_dist_torch,
                occurrence_map_torch,
                pred_torch,
            ) = model.push_forward(x)
            # pred_torch = logits.softmax(dim=1)

        image = x.detach().cpu().numpy()  # shape (B, 3, Ho, Wo)
        filename = path  # shape (B) 
        
        for characteristic_idx in range(num_characteristics):
            proto_class_identity = np.argmax(model.prototype_class_identity[characteristic_idx].cpu().numpy(), axis=1)  # shape (P)
            # record down batch data as numpy arrays
            gt = y[characteristic_idx].detach().cpu().numpy()
            protoL_input = protoL_input_torch[characteristic_idx].detach().cpu().numpy()
            proto_dist = proto_dist_torch[characteristic_idx].detach().cpu().numpy()
            occurrence_map = occurrence_map_torch[characteristic_idx].detach().cpu().numpy()
            pred = pred_torch[characteristic_idx].detach().cpu().numpy()

            # for each prototype, find the minimum distance and their indices
            for prototype_idx in range(P):
                proto_dist_j = proto_dist[:, prototype_idx]  # (B)
                if proto_class_specific[prototype_idx]:
                    # compare with only the images of the prototype's class
                    proto_dist_j = np.ma.masked_array(proto_dist_j, gt != proto_class_identity[prototype_idx])
                    if proto_dist_j.mask.all():
                        # if none of the classes this batch are the class of interest, move on
                        continue
                proto_dist_j_min = np.amin(proto_dist_j)  # scalar

                # if the distance this batch is smaller than prev.best, save it
                if proto_dist_j_min <= proto_dist_[characteristic_idx][prototype_idx]:
                    a = np.argmin(proto_dist_j)
                    
                    proto_dist_[characteristic_idx][prototype_idx] = proto_dist_j_min
                    protoL_input_[characteristic_idx][prototype_idx] = protoL_input[a, prototype_idx]
                    occurrence_map_[characteristic_idx][prototype_idx] = occurrence_map[a, prototype_idx]
                    pred_[characteristic_idx][prototype_idx] = pred[a]
                    image_[characteristic_idx][prototype_idx] = image[a]
                    gt_[characteristic_idx][prototype_idx] = gt[a]
                    filename_[characteristic_idx][prototype_idx] = filename[a]

    prototypes_similarity_to_src_ROIs = 1 - np.array(proto_dist_)  # invert distance to similarity  shape (P)
    prototypes_occurrence_maps = np.array(occurrence_map_)  # shape (P, 1, H, W)
    prototypes_src_imgs = np.array(image_)  # shape (P, 3, Ho, Wo)
    prototypes_gts = np.array(gt_)  # shape (P)
    prototypes_preds = np.array(pred_)  # shape (P, K)
    prototypes_filenames = np.array(filename_)  # shape (P)

    # save the prototype information in a pickle file
    prototype_data_dict = {
        "prototypes_filenames": prototypes_filenames,
        "prototypes_src_imgs": prototypes_src_imgs,
        "prototypes_gts": prototypes_gts,
        "prototypes_preds": prototypes_preds,
        "prototypes_occurrence_maps": prototypes_occurrence_maps,
        "prototypes_similarity_to_src_ROIs": prototypes_similarity_to_src_ROIs,
    }
    save_pickle(prototype_data_dict, f"{proto_epoch_dir}/prototypes_info.pickle")

    if replace_prototypes:
        protoL_input_ = np.array(protoL_input_)
        print("\tExecuting push ...")
        
        for idx, (prototype_vectors, protoL_input_char) in enumerate(zip(model.prototype_vectors, protoL_input_)):
            prototype_update = np.reshape(protoL_input_char, prototype_vectors.shape)
            with torch.no_grad():
                prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).to(device))
