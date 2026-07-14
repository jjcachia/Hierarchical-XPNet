import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from scipy.stats import norm


##############################################################################################################################################################
################################################################      Training Functions      ################################################################
##############################################################################################################################################################

def _train_or_test(model, data_loader, optimizer, device, is_train=True, task_weights=None, use_slice_weights=False, indeterminate=False):
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    num_tasks = model.num_tasks
    
    total_loss = 0
    
    task_losses = [0] * num_tasks  # Assuming 5 tasks

    final_pred_targets = [[] for _ in range(num_tasks)]
    final_pred_outputs = [[] for _ in range(num_tasks)]
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, _, _, _, _ in tqdm(data_loader, leave=False):  # Assuming final_target is for the final output
            X = X.to(device)
            bweights_chars = [b.float().to(device) for b in bweights_chars]
            targets = [t.long().to(device) for t in targets]
            
            final_output, task_outputs = model(X)
            
            loss = 0
            for i, (task_output, target, bweight_char) in enumerate(zip(task_outputs, targets, bweights_chars)):
                # Compute loss for each task
                if use_slice_weights:
                    bweight_char = bweight_char[0][target] # Get the first element of the batch
                    bweight_char = bweight_char * slice_weight
                    task_loss = torch.nn.functional.cross_entropy(task_output, target, reduction='none')
                    task_loss = (task_loss * bweight_char).mean()
                else:
                    bweight_char = bweight_char[0] # Get the first element of the batch
                    task_loss = torch.nn.functional.cross_entropy(task_output, target, weight=bweight_char)
                
                # Multiply the loss by the task weight
                if task_weights:
                    task_loss =  task_loss * task_weights[i]
                    
                task_losses[i] += task_loss.item()
                loss += task_loss

                # Compute accuracy for each task
                preds = task_output.argmax(dim=1)
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())   
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    average_loss = total_loss / len(data_loader)
    task_losses = [task_loss / len(data_loader) for task_loss in task_losses]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_balanced_accuracy = np.mean(task_balanced_accuracies)
    
    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_losses': task_losses,
               'task_balanced_accuracies': task_balanced_accuracies,
               'final_balanced_accuracy': final_balanced_accuracy
            }
    
    return metrics

def train_step(model, data_loader, optimizer, device, task_weights=None):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        data_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device to be used for training.
        task_weights (list): The weights for each task in the model.

    Returns:
        tuple: A tuple containing the training metrics and updated task weights.
    """
    train_metrics = _train_or_test(
        model, data_loader, optimizer, device, is_train=True, task_weights=task_weights
    )
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    for i, (loss, bal_acc) in enumerate(zip(train_metrics['task_losses'], train_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Loss: {loss:.2f}, Train Balanced Accuracy: {bal_acc*100:.2f}%")
    return train_metrics


def test_step(model, data_loader, device, task_weights=None):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        device (torch.device): The device to run the evaluation on.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    test_metrics = _train_or_test(
        model, data_loader, None, device, is_train=False, task_weights=task_weights
    )
    print(f"Val loss: {test_metrics['average_loss']:.5f}")
    for i, (loss, bal_acc) in enumerate(zip(test_metrics['task_losses'], test_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Loss: {loss:.2f}, Train Balanced Accuracy: {bal_acc*100:.2f}%")
    return test_metrics

##############################################################################################################################################################
###############################################################      Evaluation Functions      ###############################################################
##############################################################################################################################################################

# Function to evaluate the model on the test set
def evaluate_model(data_loader, model, device):
    model.eval() 
    
    final_pred_targets = [[] for _ in range(model.num_tasks)]
    final_pred_outputs = [[] for _ in range(model.num_tasks)]
    
    confusion_matrix = np.zeros((model.num_tasks, model.num_tasks), dtype=int)
    with torch.no_grad():
        for X, targets, _, _, _, _, _ in tqdm(data_loader, leave=False):
            X = X.to(device)
            targets = [t.long().to(device) for t in targets]
            
            _, task_outputs = model(X)
            
            for i, (task_output, target) in enumerate(zip(task_outputs, targets)):
                preds = task_output.argmax(dim=1)
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
                
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_balanced_accuracy = np.mean(task_balanced_accuracies)
    confusion_matrix = confusion_matrix(final_pred_targets, final_pred_outputs)
    
    metrics = {'task_balanced_accuracies': task_balanced_accuracies,
               'final_balanced_accuracy': final_balanced_accuracy
            }
    return metrics, confusion_matrix

def evaluate_model_by_nodule(model, data_loader, device, mode="median", decision_threshold=0.5, std_dev=1.2):
    model.to(device)
    model.eval()
    
    final_pred_targets = [[] for _ in range(model.num_tasks)]
    final_pred_outputs = [[] for _ in range(model.num_tasks)]
    
    with torch.no_grad():
        for slices, task_labels, labels in tqdm(data_loader, leave=False):
            slices = slices.to(device)
            
            if slices.dim() == 5:
                slices = slices.view(-1, slices.size(2), slices.size(3), slices.size(4))  # Flatten the slices into one batch
            
            _, task_outputs = model(slices)
            
            for i, (task_output, task_label) in enumerate(zip(task_outputs, task_labels)):
                if task_output.ndim > 1 and task_output.shape[1] == 1:  # If model outputs a single probability per slice
                    task_output = task_output.squeeze(1)

                if mode == "median":
                    # Calculate the median prediction for the nodule
                    task_output = torch.softmax(task_output, dim=1)
                    task_output = torch.median(task_output, dim=0).values
                
                if mode == "gaussian":
                    # Calculate the gaussian weighted average prediction for the nodule
                    task_output = torch.softmax(task_output, dim=1)
                    num_slices = task_output.size(0)
                    x = np.linspace(0, num_slices-1, num_slices)
                    mean = num_slices / 2
                    std_dev = std_dev
                    weights = norm.pdf(x, mean, std_dev)
                    weights = torch.tensor(weights, dtype=torch.float32, device=device)
                    weights = weights / weights.sum()
                    weights = weights.view(-1, 1)
                    task_output = (task_output * weights).sum(dim=0)

                preds = task_output.argmax().unsqueeze(0)
                final_pred_targets[i].extend(task_label.numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
            
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    balanced_accuracy = np.mean(task_balanced_accuracies)
    confusion_matrices = []
    for targets, predictions in zip(final_pred_targets, final_pred_outputs):
        cm = confusion_matrix(targets, predictions)
        confusion_matrices.append(cm)
    
    metrics = {'final_balanced_accuracy': balanced_accuracy,
               'task_balanced_accuracies': task_balanced_accuracies
            }

    return metrics, confusion_matrix