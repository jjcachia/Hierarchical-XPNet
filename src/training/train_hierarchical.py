import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from scipy.stats import norm

def _adjust_weights(balanced_accuracies, exponent=5, target_sum=5):
    """
    Adjusts the weights based on the balanced accuracies.

    Args:
        balanced_accuracies (list): A list of balanced accuracies.
        exponent (int, optional): The exponent used for calculating the weights. Defaults to 5.
        target_sum (int, optional): The target sum of the scaled weights. Defaults to 2.

    Returns:
        list: A list of scaled weights.
    """
    # Calculate weights as the exponentiation of the inverse of the accuracies
    weights = [1.0 / (acc ** exponent + 1e-6) for acc in balanced_accuracies]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    # Scale the normalized weights so that their sum equals the target_sum
    scaled_weights = [w * target_sum for w in normalized_weights]
    return scaled_weights


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
    
    task_losses = [0] * num_tasks

    final_pred_targets = [[] for _ in range(num_tasks)]
    final_pred_outputs = [[] for _ in range(num_tasks)]
    
    final_targets = []
    final_outputs = []
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, final_target, bweight, slice_weight, _ in tqdm(data_loader, leave=False):  # Assuming final_target is for the final output
            X = X.to(device)
            bweights_chars = [b.float().to(device) for b in bweights_chars]
            targets = [t.long().to(device) for t in targets]
            
            if indeterminate:
                final_target = final_target.long().to(device)
                bweight = bweight.float().to(device)
            else:
                final_target = final_target.float().unsqueeze(1).to(device)
                bweight = bweight.float().unsqueeze(1).to(device)
            
            if use_slice_weights:
                slice_weight = slice_weight.float().unsqueeze(1).to(device)
                bweight_pred = bweight * slice_weight
            
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

            # Compute loss for final output
            if indeterminate:
                bweight = bweight[0]
                final_loss = torch.nn.functional.cross_entropy(final_output, final_target, weight=bweight)
                final_preds = final_output.argmax(dim=1)
            else:
                final_loss = torch.nn.functional.binary_cross_entropy(final_output, final_target, weight=bweight)
                final_preds = final_output.round()
            
            loss += final_loss
            
            # Compute accuracy for final output
            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(final_preds.detach().cpu().numpy())

            total_loss += loss.item()  # Sum up total loss
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    average_loss = total_loss / len(data_loader)
    task_losses = [task_loss / len(data_loader) for task_loss in task_losses]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    
    if indeterminate:
        final_f1 = f1_score(final_targets, final_outputs, average='macro')
        final_precision = precision_score(final_targets, final_outputs, average='macro')
        final_recall = recall_score(final_targets, final_outputs, average='macro')
        final_auc = 0 
    else:
        final_f1 = f1_score(final_targets, final_outputs)
        final_precision = precision_score(final_targets, final_outputs)
        final_recall = recall_score(final_targets, final_outputs)
        final_auc = roc_auc_score(final_targets, final_outputs)
    
    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_losses': task_losses,
               'task_balanced_accuracies': task_balanced_accuracies,
               'final_balanced_accuracy': final_balanced_accuracy,
               'final_f1': final_f1,
               'final_precision': final_precision,
               'final_recall': final_recall,
               'final_auc': final_auc
            }
    
    return metrics

def train_step(model, data_loader, optimizer, device, task_weights=None):
    
    train_metrics = _train_or_test(
        model, data_loader, optimizer, device, is_train=True, task_weights=task_weights
    )
    
    print(f"Train loss: {train_metrics['average_loss']:.5f}")
    for i, (loss, bal_acc) in enumerate(zip(train_metrics['task_losses'], train_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Loss: {loss:.2f}, Train Balanced Accuracy: {bal_acc*100:.2f}%")

    print(f"Final Output - Train Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%, Train F1: {train_metrics['final_f1']*100:.2f}%, Recall: {train_metrics['final_recall']*100:.2f}%, Precision: {train_metrics['final_precision']*100:.2f}%")
    return train_metrics


def test_step(model, data_loader, device, task_weights=None):

    test_metrics = _train_or_test(
        model, data_loader, None, device, is_train=False, task_weights=task_weights
    )
    
    print(f"Val loss: {test_metrics['average_loss']:.5f}")
    for i, (loss, bal_acc) in enumerate(zip(test_metrics['task_losses'], test_metrics['task_balanced_accuracies']), 1):
        print(f"Task {i} - Loss: {loss:.2f}, Train Balanced Accuracy: {bal_acc*100:.2f}%")

    print(f"Final Output - Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%, F1: {test_metrics['final_f1']*100:.2f}%, Recall: {test_metrics['final_recall']*100:.2f}%, Precision: {test_metrics['final_precision']*100:.2f}%")
    return test_metrics


##############################################################################################################################################################
###############################################################      Evaluation Functions      ###############################################################
##############################################################################################################################################################

def evaluate_model(data_loader, model, device, indeterminate=False):
    model.eval()  # Set the model to evaluation mode
    
    final_pred_targets = [[] for _ in range(model.num_tasks)]
    final_pred_outputs = [[] for _ in range(model.num_tasks)]
    
    final_targets = []
    final_outputs = []
    
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for X, targets, _, final_target, _, _, _ in tqdm(data_loader, leave=False):  # Assuming final_target is for the final output
            X = X.to(device)
            targets = [t.long().to(device) for t in targets]
            
            if indeterminate:
                final_target = final_target.long().to(device)
            else:
                final_target = final_target.float().unsqueeze(1).to(device)
            
            final_output, task_outputs = model(X)
            
            for i, (task_output, target) in enumerate(zip(task_outputs, targets)):
                preds = task_output.argmax(dim=1)
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
            
            if indeterminate:
                final_preds = final_output.argmax(dim=1)
            else:
                final_preds = final_output.round()

            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(preds.detach().cpu().numpy())
    
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    final_balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    final_f1 = f1_score(final_targets, final_outputs)
    final_precision = precision_score(final_targets, final_outputs)
    final_recall = recall_score(final_targets, final_outputs)
    final_auc = roc_auc_score(final_targets, final_outputs)
    conf_matrix = confusion_matrix(final_targets, final_outputs)
    
    metrics = {'task_balanced_accuracies': task_balanced_accuracies,
               'final_balanced_accuracy': final_balanced_accuracy,
               'final_f1': final_f1,
               'final_precision': final_precision,
               'final_recall': final_recall,
               'final_auc': final_auc
            }
    
    return metrics, conf_matrix


def evaluate_model_by_nodule(model, data_loader, device, mode="median", decision_threshold=0.5, std_dev=1.2):
    model.to(device)
    model.eval()
    
    final_pred_targets = [[] for _ in range(model.num_tasks)]
    final_pred_outputs = [[] for _ in range(model.num_tasks)]
    
    final_targets = []
    final_outputs = []
    
    with torch.no_grad():
        for slices, task_labels, labels in tqdm(data_loader, leave=False):
            slices = slices.to(device)
            
            # Reshape slices if your model expects a single batch dimension
            if slices.dim() == 5:
                slices = slices.view(-1, slices.size(2), slices.size(3), slices.size(4))  # Flatten the slices into one batch
            
            outputs, task_outputs = model(slices)
            
            for i, (task_output, task_label) in enumerate(zip(task_outputs, task_labels)):
                if task_output.ndim > 1 and task_output.shape[1] == 1:
                    task_output = task_output.squeeze(1)

                task_output = torch.softmax(task_output, dim=1)
                
                if mode == "median":
                    task_output = torch.median(task_output, dim=0).values
                
                if mode == "gaussian":
                    num_slices = task_output.size(0)
                    x = np.linspace(0, num_slices-1, num_slices)
                    mean = num_slices / 2
                    weights = norm.pdf(x, mean, std_dev)
                    weights = torch.tensor(weights, dtype=torch.float32, device=device)
                    weights = weights / weights.sum()
                    weights = weights.view(-1, 1)
                    task_output = (task_output * weights).sum(dim=0)

                preds = task_output.argmax().unsqueeze(0)
                final_pred_targets[i].extend(task_label.numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
            
            if outputs.ndim > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)

            if mode == "median":
                outputs = outputs.median()
            if mode == "gaussian":
                num_slices = outputs.size(0)
                x = np.linspace(0, num_slices-1, num_slices)
                mean = num_slices / 2
                weights = norm.pdf(x, mean, std_dev)
                weights = torch.tensor(weights, dtype=torch.float32, device=device)
                weights = weights / weights.sum()
                outputs = (outputs * weights).sum()

            predictions = (outputs > decision_threshold).float()

            final_targets.append(labels.numpy())
            final_outputs.append(predictions.cpu().numpy())

    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(final_pred_targets, final_pred_outputs)]
    balanced_accuracy = balanced_accuracy_score(final_targets, final_outputs)
    f1 = f1_score(final_targets, final_outputs)
    precision = precision_score(final_targets, final_outputs)
    recall = recall_score(final_targets, final_outputs)
    auc = roc_auc_score(final_targets, final_outputs)
    conf_matrix = confusion_matrix(final_targets, final_outputs)
    
    metrics = {'final_balanced_accuracy': balanced_accuracy,
               'final_f1': f1,
               'final_precision': precision,
               'final_recall': recall,
               'final_auc': auc,
               'task_balanced_accuracies': task_balanced_accuracies
            }

    return metrics, conf_matrix