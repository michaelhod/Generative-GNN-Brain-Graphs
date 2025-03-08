import gc
import torch
import tempfile
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

from src.models.stp_gsr import STPGSR
from src.models.direct_sr import DirectSR
from src.plot_utils import (
    plot_grad_flow, 
    plot_adj_matrices, 
    create_gif_grad, 
    create_gif_adj,
    plot_losses,
)

from src.dual_graph_utils import *

from src.dual_graph_utils import revert_dual

from src.models.discriminator import Discriminator

from src.debug_utils import debug_shapes, inspect_tensor


def load_model(config):
    if config.model.name == 'stp_gsr':
        return STPGSR(config)
    elif config.model.name == 'direct_sr':
        return DirectSR(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model.name}")
    

def eval(config, model, source_data, target_data, criterion):
    n_target_nodes = config.dataset.n_target_nodes  # n_t
    
    model.eval()

    eval_output = []

    eval_loss = []

    with torch.no_grad():
        for source, target in zip(source_data, target_data):
            source_g = source['pyg']    
            target_m = target['mat']    # (n_t, n_t)

            model_pred = model(source_g, target_m) 

            if config.model.name == 'stp_gsr':
                pred_m = revert_dual(model_pred, n_target_nodes)    # (n_t, n_t)
                pred_m = pred_m.cpu().numpy()
            else:
                pred_m = model_pred.cpu().numpy()

            eval_output.append(pred_m)

            t_loss = criterion(model_pred, model_target)

            eval_loss.append(t_loss) 

    eval_loss = torch.stack(eval_loss).mean().item()

    model.train()

    return eval_output, eval_loss


# def train(config, 
#           source_data_train, 
#           target_data_train, 
#           source_data_val, 
#           target_data_val,
#           res_dir):
#     n_target_nodes = config.dataset.n_target_nodes  # n_t

#     # Initialize model, optmizer, and loss function
#     generator = load_model(config)
#     discriminator = Discriminator(n_target_nodes, config.experiment.discriminator.hidden_dim)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     generator = generator.to(device)
#     discriminator = discriminator.to(device)
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.experiment.lr)
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.experiment.lr)


#     criterion_L1 = torch.nn.L1Loss()  
#     criterion_BCE = torch.nn.BCELoss()

#     train_losses_G = []
#     train_losses_D = []
#     val_losses = []
 

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         generator.train()
#         discriminator.train()
#         step_counter = 0

#         for epoch in range(config.experiment.n_epochs):
#             batch_counter = 0
#             epoch_loss_G = 0.0
#             epoch_loss_D = 0.0

#             # Shuffle training data
#             random_idx = torch.randperm(len(source_data_train))
#             source_train = [source_data_train[i] for i in random_idx]
#             target_train = [target_data_train[i] for i in random_idx]

#             # Iteratively train on each sample. 
#             # (Using single sample training and gradient accummulation as the baseline IMANGraphNet model is memory intensive)
#             for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):
#                 source_g = source['pyg'].to(device)
#                 source_m = source['mat'].to(device)  # (n_s, n_s)
#                 target_m = target['mat'].to(device)  # (n_t, n_t)

#                 fake_adj = generator(source_g, target_m)[0]

#                 optimizer_D.zero_grad()
#                 real_output = discriminator(target_m)
#                 real_labels = torch.ones_like(real_output)
#                 fake_output = discriminator(fake_adj.detach())
#                 fake_labels = torch.zeros_like(fake_output)

#                 loss_D_real = criterion_BCE(real_output, real_labels)
#                 loss_D_fake = criterion_BCE(fake_output, fake_labels)
#                 loss_D = loss_D_real + loss_D_fake
#                 loss_D.backward()
#                 optimizer_D.step()

#                 optimizer_G.zero_grad()
#                 fake_output = discriminator(fake_adj)  # No detach() here
#                 loss_G_L1 = criterion_L1(fake_adj, target_m)
#                 loss_G_adv = criterion_BCE(fake_output, real_labels)
#                 loss_G = loss_G_adv + 100 * loss_G_L1  
#                 loss_G.backward()
#                 optimizer_G.step()

#                 epoch_loss_G += loss_G.item()
#                 epoch_loss_D += loss_D.item()
#                 batch_counter += 1

#                 # Log progress and do mini-batch gradient descent
#                 # if batch_counter % config.experiment.batch_size == 0 or batch_counter == len(source_train):
#                 #     # Log gradients for this iteration
#                 #     #plot_grad_flow(model.named_parameters(), step_counter, tmp_dir)


#                 #     # Predicetd and target matrices for plotting
#                 #     pred_plot = model_pred.detach()
#                 #     target_plot = model_target.detach()

#                 #     # Convert edge features to adjacency matrices
#                 #     if config.model.name == 'stp_gsr':
#                 #         pred_plot = revert_dual(pred_plot, n_target_nodes) # (n_t, n_t)
#                 #         target_plot = revert_dual(target_plot, n_target_nodes) # (n_t, n_t)

#                 #     pred_plot_m = pred_plot.cpu().numpy()
#                 #     target_plot_m = target_plot.cpu().numpy()

#                 #     # Log source, target, and predicted adjacency matrices for this iteration
#                 #     # plot_adj_matrices(source_m.detach().cpu(), pred_plot_m, target_plot_m, step_counter, tmp_dir)
                    
#                 #     # Perform gradient descent
#                 #     optimizer.step()
#                 #     optimizer.zero_grad()

#                 #     step_counter += 1

#                 #     torch.cuda.empty_cache()
#                 #     gc.collect()

#             epoch_loss_G /= len(source_data_train)
#             epoch_loss_D /= len(source_data_train)
#             train_losses_G.append(epoch_loss_G)
#             train_losses_D.append(epoch_loss_D)
#             print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Generator Loss: {epoch_loss_G}, Discriminator Loss: {epoch_loss_D}")

#             if config.experiment.log_val_loss:
#                 val_loss = eval(config, generator, source_data_val, target_data_val, criterion_L1)
#                 val_losses.append(val_loss)
#                 print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Val Loss: {val_loss}")


   

#         # Save and plot losses
#         torch.save(generator.state_dict(), f"{res_dir}/generator.pth")
#         torch.save(discriminator.state_dict(), f"{res_dir}/discriminator.pth")
#         np.save(f'{res_dir}/train_losses_G.npy', np.array(train_losses_G))
#         np.save(f'{res_dir}/train_losses_D.npy', np.array(train_losses_D))
#         np.save(f'{res_dir}/val_losses.npy', np.array(val_losses))
#         plot_losses(train_losses_G, 'train_losses_G', res_dir)
#         plot_losses(train_losses_D, 'train_losses_D', res_dir)
#         plot_losses(val_losses, 'val', res_dir)

#         # Create gif for gradient flows
#         #gif_path = f"{res_dir}/gradient_flow.gif"
#         #create_gif_grad(tmp_dir, gif_path)
#         #print(f"Gradient flow saved as {gif_path}")

#         # Create gif for training samples
#         #gif_path = f"{res_dir}/train_samples.gif"
#         #create_gif_adj(tmp_dir, gif_path)
#         #print(f"Training samples saved as {gif_path}")

    

#     return {
#         'model': model,
#         'criterion': criterion,
#     }

def train(config, 
          source_data_train, 
          target_data_train, 
          source_data_val, 
          target_data_val,
          res_dir):
    n_target_nodes = config.dataset.n_target_nodes  # n_t

    # Initialize model, optmizer, and loss function
    generator = load_model(config)
    discriminator = Discriminator(n_target_nodes, config.experiment.discriminator.hidden_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.experiment.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.experiment.lr)

    criterion_L1 = torch.nn.L1Loss()  
    criterion_BCE = torch.nn.BCELoss()

    train_losses_G = []
    train_losses_D = []
    val_losses = []
 
    with tempfile.TemporaryDirectory() as tmp_dir:
        generator.train()
        discriminator.train()
        step_counter = 0

        for epoch in range(config.experiment.n_epochs):
            batch_counter = 0
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            # Shuffle training data
            random_idx = torch.randperm(len(source_data_train))
            source_train = [source_data_train[i] for i in random_idx]
            target_train = [target_data_train[i] for i in random_idx]

            # Iteratively train on each sample
            for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):
                source_g = source['pyg'].to(device)
                source_m = source['mat'].to(device)    # (n_s, n_s)
                target_m = target['mat'].to(device)    # (n_t, n_t)

                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Real samples
                real_output = discriminator(target_m)
                real_labels = torch.ones_like(real_output)
                loss_D_real = criterion_BCE(real_output, real_labels)
                
                # Fake samples - get dual graph features from generator
                dual_pred_x = generator(source_g, target_m)
                
                # Convert dual graph features back to adjacency matrix for discriminator
                fake_adj = revert_dual(dual_pred_x, n_target_nodes)
                
                fake_output = discriminator(fake_adj.detach())  # Detach to avoid backprop through generator
                fake_labels = torch.zeros_like(fake_output)
                loss_D_fake = criterion_BCE(fake_output, fake_labels)
                
                # Total discriminator loss
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                
                # Get dual features for target (ground truth)
                dual_target_x = create_dual_graph_feature_matrix(target_m)
                
                # L1 loss on dual graph features
                loss_G_L1 = criterion_L1(dual_pred_x, dual_target_x)
                
                # Adversarial loss (with non-detached fake_adj)
                fake_output = discriminator(fake_adj)  # Note: not detached here
                loss_G_adv = criterion_BCE(fake_output, real_labels)  # Generator wants discriminator to think fake is real
                
                # Total generator loss
                loss_G = loss_G_L1 + loss_G_adv
                loss_G.backward()
                optimizer_G.step()

                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()
                batch_counter += 1

            epoch_loss_G /= len(source_data_train)
            epoch_loss_D /= len(source_data_train)
            train_losses_G.append(epoch_loss_G)
            train_losses_D.append(epoch_loss_D)
            print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Generator Loss: {epoch_loss_G}, Discriminator Loss: {epoch_loss_D}")

            if config.experiment.log_val_loss:
                # Validation function needs to be updated to work with STPGSR model
                val_loss = eval_stpgsr(config, generator, source_data_val, target_data_val, criterion_L1)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}/{config.experiment.n_epochs}, Val Loss: {val_loss}")

        # Save and plot losses
        torch.save(generator.state_dict(), f"{res_dir}/generator.pth")
        torch.save(discriminator.state_dict(), f"{res_dir}/discriminator.pth")
        np.save(f'{res_dir}/train_losses_G.npy', np.array(train_losses_G))
        np.save(f'{res_dir}/train_losses_D.npy', np.array(train_losses_D))
        np.save(f'{res_dir}/val_losses.npy', np.array(val_losses))
        plot_losses(train_losses_G, 'train_losses_G', res_dir)
        plot_losses(train_losses_D, 'train_losses_D', res_dir)
        plot_losses(val_losses, 'val', res_dir)

    return {
        'generator': generator,
        'discriminator': discriminator,
    }

def eval_stpgsr(config, generator, source_data_val, target_data_val, criterion):
    """
    Evaluation function for STPGSR model
    """
    device = next(generator.parameters()).device
    generator.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for source, target in zip(source_data_val, target_data_val):
            source_g = source['pyg'].to(device)
            target_m = target['mat'].to(device)
            
            # Get predictions
            dual_pred_x = generator(source_g, target_m)
            
            # Get target dual features
            dual_target_x = create_dual_graph_feature_matrix(target_m)
            
            # Calculate loss
            loss = criterion(dual_pred_x, dual_target_x)
            total_loss += loss.item()
    
    generator.train()
    return total_loss / len(source_data_val)