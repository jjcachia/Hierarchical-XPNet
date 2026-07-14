import os, time, gc, argparse, shutil
import pandas as pd
import torch, torch.utils.data, torchvision.transforms as transforms, torch.nn as nn
import numpy as np

from src.utils.helpers import save_metrics_to_csv, plot_and_save_loss, save_model_in_chunks, setup_directories, load_model_from_chunks, set_seed

from src.models.base_model import construct_baseModel
from src.models.baseline_model import construct_baselineModel

CHOSEN_CHARS = [True, True, False, True, True, False, False, True] # [diameter, subtlety, calcification, sphericity, margin, lobulation, spiculation, texture]

DEFAULT_BATCH_SIZE = 25
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.00001

MODEL_DICT = {
    'baseline': construct_baselineModel,
    'base': construct_baseModel
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model on the specified dataset.")
    parser.add_argument('--experiment_run', type=str, required=True, help='Identifier for the experiment run')
    
    parser.add_argument('--backbone', type=str, default='denseNet121', help='Feature Extractor Backbone to use')
    parser.add_argument('--model', type=str, default='base', help='Model to train')
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
    plot_path = os.path.join(paths['plots'], 'loss_plot.png')

    # Set the seed for reproducibility
    set_seed(27)
    
    # Save the script to the experiment directory
    shutil.copy(__file__, os.path.join(paths['scripts'], 'main.py'))
    
    # Load the model training functions    
    if args.model == 'base':
        from src.training.train_final_prediction import train_step, test_step, evaluate_model, evaluate_model_by_nodule
    elif args.model == 'baseline':
        from src.training.train_hierarchical import train_step, test_step, evaluate_model, evaluate_model_by_nodule
    
    # Check if CUDA is available
    print("#"*100 + "\n\n")
    if torch.cuda.is_available():
        print("CUDA is available. GPU devices:")
        # Loop through all available GPUs
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Only CPU is available.")
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:' + args.device) 
    print(f"Using device: {device}")

    ###############################################################################################################
    #################################### Initialize the data loaders ##############################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    # Load the labels file
    labels_file = os.path.join(script_dir, 'dataset', '2D', 'Meta', 'central_slice_labels.csv')
    
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
    ###################################### Initialize the model ###################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    epochs = 100

    if args.model not in MODEL_DICT:
        raise ValueError(f"Unsupported model name {args.model}")
    construct_Model = MODEL_DICT[args.model]
    if args.weights == 'None':
        args.weights = None
    
    # Create the model instance
    model = construct_Model(
        backbone_name=args.backbone, 
        weights=args.weights, 
        num_tasks=sum(CHOSEN_CHARS), 
        num_classes=args.classes,
        indeterminate=args.indeterminate
    )
    
    if args.model == 'base':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer_specs = [{'params': model.backbone.parameters(), 'lr': args.learning_rate/2},
                            {'params': model.task_specific_layers.parameters(), 'lr': args.learning_rate},
                            {'params': model.task_specific_classifier.parameters(), 'lr': args.learning_rate},
                            {'params': model.final_classifier.parameters(), 'lr': args.learning_rate}]
        optimizer = torch.optim.Adam(optimizer_specs)
    

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    
    ###############################################################################################################
    ####################################### Training the model ####################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    # Initialize lists to store metrics over epochs
    all_train_metrics = []
    all_test_metrics = []

    # Train the model
    start_time = time.time()  # Record the start time of the entire training
    max_val_bacc = float(0)
    for epoch in range(epochs):
        # Print header
        print("\n" + "-"*100 + f"\nEpoch: {epoch + 1}/{epochs},\t\n" + "-"*100)# + f"Task Weights: {[f'{weight:.2f}' for weight in task_weights]}\n" +
        # Train and test the model batch by batch
        epoch_start = time.time()  # Start time of the current epoch

        # Training step
        train_metrics = train_step(data_loader=train_dataloader, 
                                    model=model, 
                                    optimizer=optimizer,
                                    device=device)
        all_train_metrics.append(train_metrics)  
        
        # Testing step
        test_metrics = test_step(data_loader=val_dataloader,
                                 model=model,
                                 device=device)
        all_test_metrics.append(test_metrics) 
        
        # Save the model if the val f1 has decreased
        if test_metrics['final_balanced_accuracy'] > max_val_bacc and test_metrics['final_balanced_accuracy'] > 0.60: # and epoch > 10:
            max_val_bacc = test_metrics['final_balanced_accuracy']
            save_model_in_chunks(model.state_dict(), best_model_path)

        epoch_end = time.time()  # End time of the current epoch
        print(f"\nEpoch {epoch + 1} completed in {epoch_end - epoch_start:.2f} seconds")  # Print the time taken for the epoch

    total_time = time.time() - start_time  # Total time for training
    print(f"Total training time: {total_time:.2f} seconds\n")  # Print the total training time

    save_metrics_to_csv(all_train_metrics, all_test_metrics, metrics_path)  # Save metrics to a CSV file
    # plot_and_save_loss(all_train_metrics, all_test_metrics, plot_path)  # Plot and save the loss
    
    ###############################################################################################################
    ####################################### Evaluate the model ####################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    # Load the best model
    model.load_state_dict(load_model_from_chunks(best_model_path))
    
    # Evaluate the model on each slice
    LIDC_testset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, split='test')
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=1, shuffle=True, num_workers=0)
    
    test_metrics, test_confusion_matrix = evaluate_model(test_dataloader, model, device)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)    
    
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
    
    # Save the test metrics to a CSV file
    df_test = pd.DataFrame([test_metrics])
    df_test.to_csv(test_metrics_path)

if __name__ == '__main__':
    main()
