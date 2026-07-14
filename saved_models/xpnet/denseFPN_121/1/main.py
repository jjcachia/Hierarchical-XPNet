import os, time, gc, argparse, shutil
from PIL import Image
import torch, torch.utils.data, torchvision.transforms as transforms, torch.nn as nn

from src.utils.helpers import save_metrics_to_csv, plot_and_save_loss, save_model_in_chunks, setup_directories
from src.loaders.dataloaderv2 import LIDCDataset
import src.training.train_xpnet as tnt
from src.models.XProtoNet import construct_XPNet
from src.training.push_xpnet import push_prototypes


DEFAULT_IMG_CHANNELS = 3
DEFAULT_IMG_SIZE = 100

DEFAULT_CHARS = [False, False, False, True, True, False, False, True, True]
DEFAULT_NUM_CHARS = sum(DEFAULT_CHARS)
DEFAULT_PROTOTYPE_SHAPE = (10*DEFAULT_NUM_CHARS*2, 128, 1, 1)

DEFAULT_BATCH_SIZE = 25
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.0001

MODEL_DICT = {
    'xpnet': construct_XPNet
}

# TODO: Add number of prototypes as a parameter, prototype size, and more 
def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model on the specified dataset.")
    parser.add_argument('--backbone', type=str, default='denseFPN_121', help='Feature Extractor Backbone to use')
    parser.add_argument('--model', type=str, default='xpnet', help='Model to train')
    parser.add_argument('--experiment_run', type=str, required=True, help='Identifier for the experiment run')
    parser.add_argument('--weights', type=str, default='DEFAULT', help='Weights to use for the backbone model')
    
    parser.add_argument('--img_channels', type=int, default=DEFAULT_IMG_CHANNELS, help='Number of channels in the input image')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help='Size of the input image')
    
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
    plot_path = os.path.join(paths['plots'], 'loss_plot.png')

    # Save the script to the experiment directory
    shutil.copy(__file__, os.path.join(paths['scripts'], 'main.py'))
    
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

    # labels_file = './dataset/Meta/meta_info_old.csv'
    labels_file = os.path.join(script_dir, 'dataset', 'Meta', 'meta_info_old.csv')

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize(256),  # First resize to larger dimensions
        transforms.CenterCrop(224),  # Then crop to 224x224
        transforms.ToTensor(),  # Convert to tensor (also scales pixel values to [0, 1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # train set
    LIDC_trainset = LIDCDataset(labels_file=labels_file, chosen_chars=DEFAULT_CHARS, auto_split=True, zero_indexed=False, 
                                                            transform=transforms.Compose([transforms.Grayscale(num_output_channels=args.img_channels), 
                                                                        transforms.Resize(size=(args.img_size, args.img_size), interpolation=Image.BILINEAR), 
                                                                        transforms.ToTensor(), 
                                                                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                                        transforms.Normalize(mean, std)
                                                                        ]),
                                                            train=True)
    train_dataloader = torch.utils.data.DataLoader(LIDC_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # push set
    LIDC_pushset = LIDCDataset(labels_file=labels_file, chosen_chars=DEFAULT_CHARS, auto_split=True, zero_indexed=False, 
                                                            transform=transforms.Compose([transforms.Grayscale(num_output_channels=args.img_channels), 
                                                                        transforms.Resize(size=(args.img_size, args.img_size), interpolation=Image.BILINEAR), 
                                                                        transforms.ToTensor(), 
                                                                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                                        transforms.Normalize(mean, std)
                                                                        ]), train=True, push=True)
    push_dataloader = torch.utils.data.DataLoader(LIDC_pushset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # test set
    LIDC_testset = LIDCDataset(labels_file=labels_file, chosen_chars=DEFAULT_CHARS, auto_split=True, zero_indexed=False, 
                                                            transform=transforms.Compose([transforms.Grayscale(num_output_channels=args.img_channels), 
                                                                        transforms.Resize(size=(args.img_size, args.img_size), interpolation=Image.BILINEAR), 
                                                                        transforms.ToTensor(), 
                                                                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                                                        transforms.Normalize(mean, std)
                                                                        ]),
                                                            train=False)
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    batch_images = next(iter(train_dataloader))

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
        img_size=args.img_size,
        prototype_shape=DEFAULT_PROTOTYPE_SHAPE,
        num_characteristics=DEFAULT_NUM_CHARS,
        prototype_activation_function='log', 
        add_on_layers_type='regular'
    )
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", total_params)
    
    # Define the optimizers and learning rate schedulers
    joint_optimizer_lrs = {'features': 1e-4,
                        'add_on_layers': 1e-3,
                        'prototype_vectors': 1e-3}
    warm_optimizer_lrs = {'add_on_layers': 1e-3,
                        'prototype_vectors': 1e-3}
    last_layer_optimizer_lr = 1e-4 # 5e-4
    
    joint_optimizer_specs = \
    [{'params': model.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': model.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

    warm_optimizer_specs = \
    [{'params': model.features.adaptation_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.features.fpn.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

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
    
    prototype_activation_function = 'log'
    
    coefs = {
    'crs_ent': 1,
    'clst': 0.08,
    'sep': 0.08,
    'l1': 1e-4,
    'l1_occ': 1e-4,
    'trans' : 1e-3
    }

    # Initialize lists to store metrics over epochs
    all_train_metrics = []
    all_test_metrics = []

    # Train the model
    start_time = time.time()  # Record the start time of the entire training
    # min_test_loss = float('inf')
    task_weights = [1.0] * 4
    for epoch in range(epochs):
        # Print header
        print("\n" + "-"*100 + f"\nEpoch: {epoch + 1}/{epochs},\t" + f"Task Weights: {[f'{weight:.2f}' for weight in task_weights]}\n" + "-"*100)
        # Train and test the model batch by batch
        epoch_start = time.time()  # Start time of the current epoch
        
        if epoch < num_warm_epochs:
            tnt.warm_only(model=model)

            train_metrics, task_weights = tnt.train_xpnet(data_loader=train_dataloader, 
                                                          model=model, 
                                                          optimizer=warm_optimizer,
                                                          device=device,
                                                          coefs=coefs,
                                                          task_weights=task_weights)
            
            all_train_metrics.append(train_metrics)  # Append training metrics for the epoch
        
        else:
            tnt.joint(model=model)
            
            train_metrics, task_weights = tnt.train_xpnet(data_loader=train_dataloader, 
                                                          model=model, 
                                                          optimizer=joint_optimizer,
                                                          device=device,
                                                          coefs=coefs,
                                                          task_weights=task_weights)
            
            all_train_metrics.append(train_metrics)
        
        # Testing step
        test_metrics = tnt.test_xpnet(data_loader=test_dataloader,
                                        model=model,
                                        device=device,
                                        coefs=coefs,
                                        task_weights=task_weights)
        
        all_test_metrics.append(test_metrics)  # Append testing metrics for the epoch
        
        if epoch >= push_start and epoch in push_epochs:
            print(f"\nPushing prototypes at epoch {epoch}\n")
            push_prototypes(push_dataloader, 
                            model, 
                            device=device,
                            class_specific=True, 
                            preprocess_input_function=None, 
                            root_dir_for_saving_prototypes=paths['prototypes'],
                            epoch_number=epoch,
                            replace_prototypes=True
                        )
            
            test_metrics = tnt.test_xpnet(data_loader=test_dataloader,
                                          model=model,
                                          device=device,
                                          coefs=coefs,
                                          task_weights=task_weights)
            
            all_test_metrics.append(test_metrics)
            
            if prototype_activation_function != 'linear':
                tnt.last_only(model=model)
                for i in range(10):
                    _, task_weights = tnt.train_xpnet(data_loader=train_dataloader, 
                                                      model=model, 
                                                      optimizer=last_layer_optimizer,
                                                      device=device,
                                                      coefs=coefs,
                                                      task_weights=task_weights)
                    
                    _ = tnt.test_xpnet(data_loader=test_dataloader,
                                        model=model,
                                        device=device,
                                        coefs=coefs,
                                        task_weights=task_weights)
        
        # Save the model if the test loss has decreased
        # if test_metrics['average_loss'] < min_test_loss:
        #     min_test_loss = test_metrics['average_loss']
        #     save_model_in_chunks(model.state_dict(), best_model_path)

        epoch_end = time.time()  # End time of the current epoch
        print(f"\nEpoch {epoch + 1} completed in {epoch_end - epoch_start:.2f} seconds")  # Print the time taken for the epoch

    total_time = time.time() - start_time  # Total time for training
    print(f"Total training time: {total_time:.2f} seconds\n")  # Print the total training time

    save_metrics_to_csv(all_train_metrics, all_test_metrics, metrics_path)  # Save metrics to a CSV file
    plot_and_save_loss(all_train_metrics, all_test_metrics, plot_path)  # Plot and save the loss

if __name__ == '__main__':
    main()
