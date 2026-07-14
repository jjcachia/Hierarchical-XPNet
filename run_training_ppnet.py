import os, time, gc, argparse, shutil
import pandas as pd
import numpy as np
from PIL import Image
import torch, torch.utils.data, torchvision.transforms as transforms, torch.nn as nn

from src.utils.helpers import save_metrics_to_csv, plot_and_save_loss, save_model_in_chunks, setup_directories, load_model_from_chunks, set_seed

from src.models.ProtoPNet import construct_PPNet
import src.training.train_ppnet as tnt
from src.training.push import push_prototypes

CHOSEN_CHARS = [True, True, False, True, True, False, False, True] # [diameter, subtlety, calcification, sphericity, margin, lobulation, spiculation, texture]
DEFAULT_NUM_CHARS = sum(CHOSEN_CHARS)
DEFAULT_NUM_CLASSES = 2
DEFAULT_NUM_PROTOTYPES_PER_CLASS = 10
DEFAULT_PROTOTYPE_SHAPE = (DEFAULT_NUM_PROTOTYPES_PER_CLASS*DEFAULT_NUM_CHARS*DEFAULT_NUM_CLASSES, 128, 1, 1)
# DEFAULT_PROTOTYPE_SHAPE = (10*2*DEFAULT_NUM_CHARS, 128, 1, 1)

DEFAULT_BATCH_SIZE = 25
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.0001

MODEL_DICT = {
    'ppnet': construct_PPNet
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model on the specified dataset.")
    parser.add_argument('--experiment_run', type=str, required=True, help='Identifier for the experiment run')
    
    parser.add_argument('--backbone', type=str, default='denseNet121', help='Feature Extractor Backbone to use')
    parser.add_argument('--model', type=str, default='ppnet', help='Model to train')
    parser.add_argument('--weights', type=str, default='DEFAULT', help='Weights to use for the backbone model')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes to predict')
    parser.add_argument('--indeterminate', type=bool, default=False, help='Whether to predict indeterminate nodules')
    
    parser.add_argument('--device', type=str, default='0', help='GPU device to use')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for optimizer')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define your experiment details
    base_path = os.path.join(script_dir, 'saved_models')
    experiment_model = args.model
    experiment_backbone = args.backbone
    experiment_run = args.experiment_run

    # Setup directories
    paths = setup_directories(base_path, experiment_model, experiment_backbone, experiment_run)
    best_model_path = os.path.join(paths['weights'], 'best_model.pth')
    metrics_path = os.path.join(paths['metrics'], 'metrics.csv')
    test_metrics_path = os.path.join(paths['metrics'], 'test_metrics.csv')

    # Prototype push path
    push_dir = paths['prototypes']
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'proto-bound-boxes'
    
    # Save the script to the experiment directory
    shutil.copy(__file__, os.path.join(paths['scripts'], 'main.py'))
    
    # Set the seed for reproducibility
    set_seed(27)
    
    # Check if CUDA is available
    print("#"*100 + "\n\n")
    if torch.cuda.is_available():
        print("CUDA is available. GPU devices:")
        # Loop through all available GPUs
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Only CPU is available.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ###############################################################################################################
    #################################### Initialize the data loaders ##############################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    # Load the labels file
    labels_file = os.path.join(script_dir, 'dataset', '2D', 'Meta', 'processed_central_slice_labels.csv')
    
    # Check if the labels file includes 3D data
    if '3D' in labels_file:
        from src.loaders._3D.dataloader import LIDCDataset
    elif '2_5D' in labels_file:
        from src.loaders._2_5D.dataloader import LIDCDataset
    elif '2D' in labels_file:
        from src.loaders._2D.dataloader import LIDCDataset
    
    # train set
    LIDC_trainset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, split='train')
    train_dataloader = torch.utils.data.DataLoader(LIDC_trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # push set
    LIDC_pushset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, split='push')
    push_dataloader = torch.utils.data.DataLoader(LIDC_pushset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # validation set
    LIDC_valset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, split='val')
    val_dataloader = torch.utils.data.DataLoader(LIDC_valset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    batch_images = next(iter(train_dataloader))

    if '3D' in labels_file:
        print(f"Batch Size: {batch_images[0].shape[0]}, Number of Channels: {batch_images[0].shape[1]}, Image Size: {batch_images[0].shape[2]} x {batch_images[0].shape[3]} x {batch_images[0].shape[4]} (NCDHW)\n")
    else:
        print(f"Batch Size: {batch_images[0].shape[0]}, Number of Channels: {batch_images[0].shape[1]}, Image Size: {batch_images[0].shape[2]} x {batch_images[0].shape[3]} (NCHW)\n")
        
    print(f"Number of Characteristics: {len(batch_images[1])}")


    ###############################################################################################################
    ##################################### Initialize the model and optimizer ######################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    if args.model not in MODEL_DICT:
        raise ValueError(f"Unsupported model name {args.model}")
    construct_Model = MODEL_DICT[args.model]
    
    # Create the model instance
    model = construct_Model(
        base_architecture=args.backbone, 
        weights=args.weights,
        img_size=100,
        prototype_shape=DEFAULT_PROTOTYPE_SHAPE,
        num_characteristics=DEFAULT_NUM_CHARS,
        num_classes=DEFAULT_NUM_CLASSES,
        prototype_activation_function='log', 
        add_on_layers_type='regular'
    )
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", total_params)
    
    # Define the optimizers and learning rate schedulers
    warm_optimizer_lrs = {'add_on_layers': 1e-3,
                        'prototype_vectors': 1e-3}
    joint_optimizer_lrs = {'features': 1e-4,
                        'add_on_layers': 1e-3,
                        'prototype_vectors': 1e-3}
    last_layer_optimizer_lr = 1e-4 # 5e-4
    
    warm_optimizer_specs = \
    [#{'params': model.features.adaptation_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    #{'params': model.features.fpn.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    
    joint_optimizer_specs = \
    [{'params': model.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': model.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

    last_layer_optimizer_specs = [{'params': model.task_specific_classifier.parameters(), 'lr': last_layer_optimizer_lr},
                                 {'params': model.final_classifier.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    

    ###############################################################################################################
    ############################################# Training the model ##############################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    # Set the number of epochs (we'll keep this small for faster training times)
    epochs = args.epochs
    num_warm_epochs = 10
    push_start = 10
    push_epochs = [i for i in range(epochs) if i % push_start == 0]
    
    coefs = {
    'crs_ent': 1,#0.4,#1.1,#0.8,#1.1,#1,#0.5,#changed from 1 at 48
    'clst': 0.8*1.5,#*1.5,#0.2,#0.3,#1.1,#0.8,#5,#0.8,
    'sep': -0.04,#-0.0004,#-0.17,#-0.22,#-0.5, #used to be -0.08 #dm made it smaller to avoid the problem I noticed where as the separation loss is subtracted, having it too large makes your loss negative making everything explode (also -0.025 works but unstable)
    'l1': 1e-4
    }

    # Initialize lists to store metrics over epochs
    all_train_metrics = []
    all_val_metrics = []

    # Train the model
    max_val_bacc = float(0)
    start_time = time.time() 
    for epoch in range(epochs):
        # Print header
        print("\n" + "-"*100 + f"\nEpoch: {epoch + 1}/{epochs},\t" + "-"*100)

        epoch_start = time.time() 
        
        # Training step
        if epoch < num_warm_epochs:
            tnt.warm_only(model=model)

            train_metrics, _ = tnt.train_ppnet(data_loader=train_dataloader, 
                                                          model=model, 
                                                          optimizer=warm_optimizer,
                                                          device=device,
                                                          coefs=coefs)
            
            all_train_metrics.append(train_metrics)
        
        else:
            tnt.joint(model=model)
            train_metrics, _ = tnt.train_ppnet(data_loader=train_dataloader, 
                                               model=model, 
                                               optimizer=joint_optimizer,
                                               device=device,
                                               coefs=coefs)
            
            all_train_metrics.append(train_metrics)
        
        
        # Testing step
        val_metrics = tnt.test_ppnet(data_loader=val_dataloader,
                                        model=model,
                                        device=device,
                                        coefs=coefs)
        
        all_val_metrics.append(val_metrics)
        
        if val_metrics['final_balanced_accuracy'] > max_val_bacc and val_metrics['final_balanced_accuracy'] > 0.60:
            max_val_bacc = val_metrics['final_balanced_accuracy']
            save_model_in_chunks(model.state_dict(), best_model_path)
        
        # Push step
        if epoch >= push_start and epoch in push_epochs:
            print(f"\nPushing prototypes at epoch {epoch}\n")
            push_prototypes(push_dataloader, 
                            model, 
                            class_specific=False,# was true 
                            preprocess_input_function=None, 
                            save_prototype_class_identity=True,
                            epoch_number=epoch,
                            root_dir_for_saving_prototypes=push_dir,
                            prototype_img_filename_prefix=prototype_img_filename_prefix,
                            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                        )
            
            val_metrics = tnt.test_ppnet(data_loader=val_dataloader,
                                          model=model,
                                          device=device,
                                          coefs=coefs)
            
            all_val_metrics.append(val_metrics)
            
            tnt.last_only(model=model)
            for i in range(10):
                train_metrics, _ = tnt.train_ppnet(data_loader=train_dataloader, 
                                                   model=model, 
                                                   optimizer=last_layer_optimizer,
                                                   device=device,
                                                   coefs=coefs)
                all_train_metrics.append(train_metrics)
                
                val_metrics = tnt.test_ppnet(data_loader=val_dataloader,
                                              model=model,
                                              device=device,
                                              coefs=coefs)
                all_val_metrics.append(val_metrics)
                
                if val_metrics['final_balanced_accuracy'] > max_val_bacc and val_metrics['final_balanced_accuracy'] > 0.60:
                    max_val_bacc = val_metrics['final_balanced_accuracy']
                    save_model_in_chunks(model.state_dict(), best_model_path)

        epoch_end = time.time()  # End time of the current epoch
        print(f"\nEpoch {epoch + 1} completed in {epoch_end - epoch_start:.2f} seconds")  

    total_time = time.time() - start_time  # Total time for training
    print(f"Total training time: {total_time:.2f} seconds\n") 

    save_metrics_to_csv(all_train_metrics, all_val_metrics, metrics_path)    
    
    ###############################################################################################################
    ################################### Evaluate the model on the test set ########################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    # Evaluate the model on each slice
    LIDC_testset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, split='test')
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    _ = tnt.test_ppnet(data_loader=test_dataloader, model=model, device=device, coefs=coefs)
    test_metrics, test_confusion_matrix = tnt.evaluate_model(test_dataloader, model, device)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix) 
    
    # Load the best model
    model.load_state_dict(load_model_from_chunks(best_model_path))
    
    # Evaluate the model on each slice
    LIDC_testset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, split='test')
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    test_metrics, test_confusion_matrix = tnt.evaluate_model(test_dataloader, model, device)
    print(f"Test Metrics, Best Model:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)  
    
    df_test = pd.DataFrame([test_metrics])
    df_test.to_csv(test_metrics_path)
    
    # Check if labels includes 'central'
    if 'central' not in labels_file:
        # Group slices by nodule and evaluate the model on each nodule
        from src.evaluation.evaluating import LIDCEvaluationDataset
        LIDC_testset = LIDCEvaluationDataset(labels_file=labels_file, indeterminate=False, chosen_chars=CHOSEN_CHARS)
        test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=1, shuffle=False, num_workers=0) # Predict one nodule at a time

        test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="median")
        print(f"Test Metrics with Median Aggregation:")
        print(test_metrics)
        print("Test Confusion Matrix:")
        print(test_confusion_matrix)

        test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=0.6)
        print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 0.6:")
        print(test_metrics)
        print("Test Confusion Matrix:")

        print(test_confusion_matrix)
        test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.0)
        print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.0:")
        print(test_metrics)
        print("Test Confusion Matrix:")
        print(test_confusion_matrix)

        test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.4)
        print(f"Test Metrics with Gaussian Aggregation and Standard Deviation of 1.4:")
        print(test_metrics)
        print("Test Confusion Matrix:")
        print(test_confusion_matrix)

if __name__ == '__main__':
    main()
