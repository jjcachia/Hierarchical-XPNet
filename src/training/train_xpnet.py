import torch
from tqdm import tqdm

from src.loss.loss import (
    CeLoss,
    ClusterRoiFeat,
    SeparationRoiFeat,
    OrthogonalityLoss,
    L_norm,
    TransformLoss,
)

from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix


def _train_or_test(model, data_loader, optimizer, device, is_train=True, use_l1_mask=True, coefs=None, task_weights=None):
    model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()
    
    num_characteristics = model.num_characteristics
    num_classes = model.num_classes
    
    # Initialize the loss functions
    CrossEntropy = CeLoss(loss_weight=coefs['crs_ent'])
    Cluster = ClusterRoiFeat(loss_weight=coefs['clst'], num_classes=num_classes)
    Separation = SeparationRoiFeat(loss_weight=coefs['sep'], num_classes=num_classes)
    # Orthogonality = OrthogonalityLoss(loss_weight=coefs['orth'], num_classes=num_classes)
    Transform = TransformLoss(loss_weight=coefs['trans'])
    L1_occ = L_norm(loss_weight=coefs['l1_occ'], mask=None, reduction="mean", p=1)
    
    # Initialize the task losses for each characteristic
    task_total_losses = [0.0] * num_characteristics
    task_cross_entropy = [0.0] * num_characteristics
    task_cluster_cost = [0.0] * num_characteristics
    task_separation_cost = [0.0] * num_characteristics
    task_l1 = [0.0] * num_characteristics
    task_occ_cost = [0.0] * num_characteristics
    task_targets_all = [[] for _ in range(num_characteristics)]
    task_predictions_all = [[] for _ in range(num_characteristics)]

    # Initialize the final output losses
    final_total_loss = 0.0
    final_targets_all = []
    final_predictions_all = []
    
    # Initialize the total loss
    total_loss = 0.0
    
    n_batches = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X, targets, bweights_chars, final_target, bweight, _, _ in tqdm(data_loader, leave=False):
            X = X.to(device)
            bweights_chars = [b.float().to(device) for b in bweights_chars]            
            # targets = [t.squeeze().to(device) for t in targets]
            targets = [t.long().to(device) for t in targets]
            final_target = final_target.float().unsqueeze(1).to(device)
            bweight = bweight.float().unsqueeze(1).to(device)
            
            final_output, task_outputs, similarities, occurrence_maps = model(X)

            ############################ Compute Losses ############################
            
            batch_loss = 0.0
            for characteristic_idx, (task_output, similarity, occurrence_map, target, bweight_char) in enumerate(zip(task_outputs, similarities, occurrence_maps, targets, bweights_chars)):
                # Get the prototype identity for the current characteristic
                prototype_char_identity = model.prototype_class_identity[characteristic_idx].to(device)
                
                # Compute cross entropy cost - to encourage the correct classification of the input
                cross_entropy_cost = CrossEntropy.compute(task_output, target)
                
                # Compute cluster cost - to encourage similarity among prototypes of the same class
                cluster_cost = Cluster.compute(similarity, target)
                
                # Compute separation cost - to encourage diversity among prototypes of different classes
                separation_cost = Separation.compute(similarity, target)
                
                # TODO: Compute Orthogonality loss - to encourage diversity among prototypes
                
                # Compute l1 regularization on task-specific classifier weights - to encourage sparsity in the weights
                l1_mask= 1 - torch.t(prototype_char_identity).to(device)
                l1 = L_norm(loss_weight=coefs['l1'], mask=l1_mask, p=1).compute(model.task_specific_classifier[characteristic_idx].weight)
                
                # Compute Occurance Map Transformation Regularization - to encourage the occurance map to be generalize better
                occ_trans_cost = Transform.compute(X, occurrence_map, model, characteristic_idx)
                
                # Compute Occurance Map L1 Regularization - to make the occruance map as small as possible to avoid covering more regions than necessary
                occ_l1 = L1_occ.compute(occurrence_map, dim=(-2, -1))
                
                occurance_map_cost = occ_trans_cost + occ_l1
                
                # Update the different task losses for each characteristic
                task_cross_entropy[characteristic_idx] += cross_entropy_cost.item()
                task_cluster_cost[characteristic_idx] += cluster_cost.item()
                task_separation_cost[characteristic_idx] += separation_cost.item()
                task_l1[characteristic_idx] += l1.item()
                task_occ_cost[characteristic_idx] += occurance_map_cost.item()
                
                # Collect the different losses for each characteristic
                task_loss = (
                    cross_entropy_cost 
                    + cluster_cost 
                    + separation_cost 
                    + l1
                    + occurance_map_cost
                )
                
                # Update the task total losses
                task_total_losses[characteristic_idx] += task_loss.item()
                
                # Apply task weights if provided
                if task_weights:
                    task_loss *= task_weights[characteristic_idx]
                
                # Update the total loss for the batch
                batch_loss += task_loss                
                
                # Collect statistics for each characteristic's prediction metrics
                preds = task_output.argmax(dim=1)
                task_targets_all[characteristic_idx].extend(target.cpu().numpy())
                task_predictions_all[characteristic_idx].extend(preds.detach().cpu().numpy())                

            # Compute binary cross entropy loss for final output
            final_loss = torch.nn.functional.binary_cross_entropy(final_output, final_target, weight=bweight)
            batch_loss += final_loss
            
            # Collect statistics for final prediction metrics
            final_total_loss += final_loss.item()
            final_preds = final_output.round()
            final_targets_all.extend(final_target.cpu().numpy())
            final_predictions_all.extend(final_preds.detach().cpu().numpy())
            
            total_loss += batch_loss.item()  # Sum up total loss
            
            # Compute gradient and do SGD step
            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
            n_batches += 1
    
    ############################ Compute Metrics ############################
    
    average_loss = total_loss / n_batches
    
    task_losses = [t / n_batches for t in task_total_losses]
    task_cross_entropy = [t / n_batches for t in task_cross_entropy]
    task_cluster_cost = [t / n_batches for t in task_cluster_cost]
    task_separation_cost = [t / n_batches for t in task_separation_cost]
    task_l1 = [t / n_batches for t in task_l1]
    task_occ_cost = [t / n_batches for t in task_occ_cost]
    task_balanced_accuracies = [balanced_accuracy_score(targets, outputs) for targets, outputs in zip(task_targets_all, task_predictions_all)]
    
    final_loss = final_total_loss / n_batches
    final_balanced_accuracy = balanced_accuracy_score(final_targets_all, final_predictions_all)
    final_f1 = f1_score(final_targets_all, final_predictions_all)
    final_precision = precision_score(final_targets_all, final_predictions_all)
    final_recall = recall_score(final_targets_all, final_predictions_all)
    final_auc = roc_auc_score(final_targets_all, final_predictions_all)

    # return the metrics as a dictionary
    metrics = {'average_loss': average_loss, 
               'task_losses': task_losses,
               'task_balanced_accuracies': task_balanced_accuracies, 
               'task_cross_entropy': task_cross_entropy,
               'task_cluster_cost': task_cluster_cost,
               'task_separation_cost': task_separation_cost,
               'task_l1': task_l1,
               'task_occ_cost': task_occ_cost,
               'final_loss': final_loss,
               'final_balanced_accuracy': final_balanced_accuracy,
               'final_f1': final_f1,
               'final_precision': final_precision,
               'final_recall': final_recall,
               'final_auc': final_auc
            }
    
    if is_train:
        task_weights = _adjust_weights(task_losses, exponent=5, target_sum=4) if task_weights else task_weights
        return metrics, task_weights
    else:
        return metrics


def train_xpnet(model, data_loader, optimizer, device, use_l1_mask=True, coefs=None, task_weights=None):
    train_metrics, task_weights = _train_or_test(model, data_loader, optimizer, device, is_train=True, use_l1_mask=use_l1_mask, coefs=coefs, task_weights=task_weights)
    print("\nFinal Train Metrics:")
    print(f"Total Loss: {train_metrics['average_loss']:.5f}")
    for i, (bal_acc, task_loss, task_ce, task_cc, task_sc, task_occ) in enumerate(zip(train_metrics['task_balanced_accuracies'], train_metrics['task_losses'], train_metrics['task_cross_entropy'], train_metrics['task_cluster_cost'], train_metrics['task_separation_cost'], train_metrics['task_occ_cost']), 1):
        print(f"Characteristic {i}      - Task Loss: {task_loss:.2f}, Cross Entropy: {task_ce:.2f}, Cluster Cost: {task_cc:.2f}, Separation Cost: {task_sc:.2f}, Occurance Map Cost: {task_occ:.2f}, Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Malignancy Prediction - Binary Cross Entropy Loss: {train_metrics['final_loss']:.2f}, Balanced Accuracy: {train_metrics['final_balanced_accuracy']*100:.2f}%, F1 Score: {train_metrics['final_f1']*100:.2f}%")
    return train_metrics, task_weights

def test_xpnet(model, data_loader, device, use_l1_mask=True, coefs=None, task_weights=None):
    test_metrics = _train_or_test(model, data_loader, None, device, is_train=False, use_l1_mask=use_l1_mask, coefs=coefs, task_weights=task_weights)
    print("\nFinal Test Metrics:")
    print(f"Total Loss: {test_metrics['average_loss']:.5f}")
    for i, (bal_acc, task_loss, task_ce, task_cc, task_sc, task_occ) in enumerate(zip(test_metrics['task_balanced_accuracies'], test_metrics['task_losses'], test_metrics['task_cross_entropy'], test_metrics['task_cluster_cost'], test_metrics['task_separation_cost'], test_metrics['task_occ_cost']), 1):
        print(f"Characteristic {i}      - Task Loss: {task_loss:.2f}, Cross Entropy: {task_ce:.2f}, Cluster Cost: {task_cc:.2f}, Separation Cost: {task_sc:.2f}, Occurance Map Cost: {task_occ:.2f}, Balanced Accuracy: {bal_acc*100:.2f}%")
    # Print the metrics for the final output
    print(f"Malignancy Prediction - Binary Cross Entropy Loss: {test_metrics['final_loss']:.2f}, Balanced Accuracy: {test_metrics['final_balanced_accuracy']*100:.2f}%, F1 Score: {test_metrics['final_f1']*100:.2f}%")
    return test_metrics


##############################################################################################################################################################
##############################################################      Gradient Modification      ###############################################################
##############################################################################################################################################################

def last_only(model):
    for p in model.cnn_backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    for p in model.occurrence_module.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = True
        
    for p in model.final_add_on_layers.parameters():
        p.requires_grad = False
    for p in model.final_classifier.parameters():
        p.requires_grad = True # was true

def warm_only(model):
    # if model.features.encoder is not None:
    #     for p in model.features.encoder.parameters():
    #         p.requires_grad = False
    #     for p in model.features.adaptation_layers.parameters():
    #         p.requires_grad = True
    #     for p in model.features.fpn.parameters():
    #         p.requires_grad = True
    # else:
    for p in model.cnn_backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    for p in model.occurrence_module.parameters():
            p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = False
        
    for p in model.final_add_on_layers.parameters():
        p.requires_grad = True
    for p in model.final_classifier.parameters():
        p.requires_grad = False
        
def joint(model):
    for p in model.cnn_backbone.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    for p in model.occurrence_module.parameters():
            p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.task_specific_classifier.parameters():
        p.requires_grad = False
    
    for p in model.final_add_on_layers.parameters():
        p.requires_grad = True
    for p in model.final_classifier.parameters():
        p.requires_grad = False
    

##############################################################################################################################################################
##################################################################    Evaluation Functions    ################################################################
##############################################################################################################################################################

def evaluate_model(data_loader, model, device, indeterminate=False):
    model.eval()  # Set the model to evaluation mode
    
    num_characteristics = model.num_characteristics
    
    final_pred_targets = [[] for _ in range(num_characteristics)]
    final_pred_outputs = [[] for _ in range(num_characteristics)]
    
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
            
            final_output, task_outputs, _, _ = model(X)
            
            for i, (task_output, target) in enumerate(zip(task_outputs, targets)):
                preds = task_output.argmax(dim=1)
                final_pred_targets[i].extend(target.cpu().numpy())
                final_pred_outputs[i].extend(preds.detach().cpu().numpy())  
            
            if indeterminate:
                final_preds = final_output.argmax(dim=1)
            else:
                final_preds = final_output.round()

            final_targets.extend(final_target.cpu().numpy())
            final_outputs.extend(final_preds.detach().cpu().numpy())
    
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