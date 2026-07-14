import os, time, gc, argparse, shutil
import pandas as pd
import torch, torch.utils.data, torchvision.transforms as transforms, torch.nn as nn

from src.utils.helpers import setup_directories, load_model_from_chunks, set_seed
from src.loaders._2_5D.dataloader import LIDCDataset
from src.models.base_model import construct_baseModel
from src.models.baseline_model import construct_baselineModel
from src.evaluation.evaluating import LIDCEvaluationDataset
from src.training.train_final_prediction import evaluate_model
from src.training.train_hierarchical import evaluate_model, evaluate_model_by_nodule

IMG_CHANNELS = 3
IMG_SIZE = 100
CHOSEN_CHARS = [True, True, False, True, True, False, False, True]

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
    
    parser.add_argument('--img_channels', type=int, default=IMG_CHANNELS, help='Number of channels in the input image')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='Size of the input image')
    
    parser.add_argument('--device', type=str, default='0', help='GPU device to use')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for optimizer')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, 'saved_models')
    
    # Define your experiment details
    experiment_model = args.model
    experiment_backbone = args.backbone
    experiment_run = args.experiment_run

    # Setup directories
    paths = setup_directories(base_path, experiment_model, experiment_backbone, experiment_run)
    best_model_path = os.path.join(paths['weights'], 'best_model.pth')
    test_metrics_path = os.path.join(paths['metrics'], 'test_metrics.csv')

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
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:' + args.device) 
    print(f"Using device: {device}")

    ###############################################################################################################
    ###################################### Initialize the model ###################################################
    ###############################################################################################################

    if args.model not in MODEL_DICT:
        raise ValueError(f"Unsupported model name {args.model}")
    construct_Model = MODEL_DICT[args.model]
    
    # Create the model instance
    model = construct_Model(backbone_name=args.backbone, weights=args.weights)
    # from efficientnet_pytorch_3d import EfficientNet3D #TODO: REMOVE IF NOT 3D
    # model = EfficientNet3D.from_name("efficientnet-b0", in_channels=1, override_params={'num_classes': 1})
    # model.to(device)
    model.to(device)
    ###############################################################################################################
    ####################################### Evaluate the model ####################################################
    ###############################################################################################################
    print("\n\n" + "#"*100 + "\n\n")
    
    model.load_state_dict(load_model_from_chunks(best_model_path))
    
    # labels_file = os.path.join(script_dir, 'dataset', '3D', 'Meta', 'volume_labels.csv')
    labels_file = os.path.join(script_dir, 'dataset', '2D', 'Meta', 'central_slice_labels.csv')
    # labels_file = os.path.join(script_dir, 'dataset', '2_5D', 'Meta', 'adjacent_slices_labels.csv')
    
    LIDC_testset = LIDCDataset(labels_file=labels_file, chosen_chars=CHOSEN_CHARS, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]), split='test')
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=50, shuffle=False, num_workers=0)
    
    test_metrics, test_confusion_matrix = evaluate_model(test_dataloader, model, device)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)  
    
    # Evaluate the model on the test set
    LIDC_testset = LIDCEvaluationDataset(labels_file=labels_file, indeterminate=False, transform=transforms.Compose([transforms.Grayscale(num_output_channels=IMG_CHANNELS), transforms.ToTensor()]), chosen_chars=CHOSEN_CHARS)
    test_dataloader = torch.utils.data.DataLoader(LIDC_testset, batch_size=1, shuffle=False, num_workers=0) 
    
    # Evaluate the model on each slice
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="median")
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)  
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.0)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix) 
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=0.6)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix) 
    
    test_metrics, test_confusion_matrix = evaluate_model_by_nodule(model, test_dataloader, device, mode="gaussian", std_dev=1.2)
    print(f"Test Metrics:")
    print(test_metrics)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix) 
    
    # Save the test metrics to a CSV file
    df_test = pd.DataFrame([test_metrics])
    df_test.to_csv(test_metrics_path)

if __name__ == '__main__':
    main()
