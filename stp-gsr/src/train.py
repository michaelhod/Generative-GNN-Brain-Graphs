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

    # Initialize model, optimizer, and loss function
    generator = load_model(config)
    discriminator = Discriminator(n_target_nodes)
    
    # Verify discriminator has sigmoid at output
    if not hasattr(discriminator, 'has_sigmoid_output'):
        print("WARNING: Discriminator may not have sigmoid output. Adding a check to ensure proper outputs.")
        original_forward = discriminator.forward
        
        def sigmoid_wrapped_forward(x):
            return torch.sigmoid(original_forward(x))
            
        discriminator.forward = sigmoid_wrapped_forward
        discriminator.has_sigmoid_output = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Use different learning rates for generator and discriminator
    # Generally, it helps to have discriminator learn slower
    lr_g = config.experiment.lr
    lr_d = config.experiment.lr * 0.5  # Half the learning rate for discriminator
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # Use binary cross entropy with logits for more stability
    criterion_L1 = torch.nn.L1Loss()
    criterion_BCE = torch.nn.BCELoss()

    train_losses_G = []
    train_losses_D = []
    val_losses = []
    
    # For tracking statistics
    d_real_accuracies = []
    d_fake_accuracies = []
    g_fool_accuracies = []
    
    # Add some debugging helpers
    def print_tensor_stats(tensor, name):
        if tensor is None:
            print(f"{name} is None!")
            return
        print(f"{name} stats: shape={tensor.shape}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, " 
              f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
    
    # Function to add noise to tensors
    def add_noise(tensor, noise_factor=0.05):
        noise = torch.randn_like(tensor) * noise_factor
        return tensor + noise
 
    with tempfile.TemporaryDirectory() as tmp_dir:
        generator.train()
        discriminator.train()
        step_counter = 0

        for epoch in range(config.experiment.n_epochs):
            batch_counter = 0
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            epoch_g_fool_acc = 0.0

            # Shuffle training data
            random_idx = torch.randperm(len(source_data_train))
            source_train = [source_data_train[i] for i in random_idx]
            target_train = [target_data_train[i] for i in random_idx]

            # Iteratively train on each sample
            for source, target in tqdm(zip(source_train, target_train), total=len(source_train)):
                source_g = source['pyg'].to(device)
                source_m = source['mat'].to(device)    # (n_s, n_s)
                target_m = target['mat'].to(device)    # (n_t, n_t)
                
                # Print statistics for first batch of first epoch
                if epoch == 0 and batch_counter == 0:
                    print_tensor_stats(target_m, "Target Adjacency Matrix")
                
                # -------------- Debug early forward pass --------------
                if epoch == 0 and batch_counter == 0:
                    with torch.no_grad():
                        print("\n===== Initial Generator Output Debug =====")
                        dual_pred_x_debug = generator(source_g, target_m)
                        print_tensor_stats(dual_pred_x_debug, "Generator Dual Output")
                        
                        fake_adj_debug = revert_dual(dual_pred_x_debug, n_target_nodes)
                        print_tensor_stats(fake_adj_debug, "Converted Adjacency Matrix")
                        
                        dual_target_x_debug = create_dual_graph_feature_matrix(target_m)
                        print_tensor_stats(dual_target_x_debug, "Target Dual Features")
                        print("=============================================\n")

                # ======== Train Discriminator ========
                optimizer_D.zero_grad()
                
                # --- Real samples with label smoothing ---
                real_target_m = add_noise(target_m, 0.05)  # Add small noise to real samples
                real_output = discriminator(real_target_m)
                # Use label smoothing: target 0.9 instead of 1.0
                real_labels = torch.ones_like(real_output) * 0.9  
                loss_D_real = criterion_BCE(real_output, real_labels)
                
                # --- Fake samples ---
                dual_pred_x = generator(source_g, target_m)
                
                # Check for NaN or extreme values
                if torch.isnan(dual_pred_x).any():
                    print(f"WARNING: NaN detected in generator output at epoch {epoch}, batch {batch_counter}")
                    dual_pred_x = torch.nan_to_num(dual_pred_x, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Convert dual graph features to adjacency matrix
                fake_adj = revert_dual(dual_pred_x, n_target_nodes)
                
                # Add noise to fake samples too
                fake_adj = add_noise(fake_adj, 0.05)
                
                # Clip to valid range if needed
                fake_adj = torch.clamp(fake_adj, 0.0, 1.0)
                
                fake_output = discriminator(fake_adj.detach())  # Detach to avoid backprop through generator
                # Label smoothing: target 0.1 instead of 0.0
                fake_labels = torch.zeros_like(fake_output) + 0.1
                loss_D_fake = criterion_BCE(fake_output, fake_labels)
                
                # Calculate discriminator accuracy for monitoring
                d_real_acc = ((real_output > 0.5).float().mean()).item()
                d_fake_acc = ((fake_output < 0.5).float().mean()).item()
                
                # Total discriminator loss
                loss_D = loss_D_real + loss_D_fake
                
                # Only update discriminator if it's not too strong
                update_D = True
                if d_real_acc > 0.95 and d_fake_acc > 0.95:
                    update_D = False
                    print(f"Skipping discriminator update at epoch {epoch}, batch {batch_counter} - discriminator too strong")
                
                if update_D:
                    loss_D.backward()
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_D.step()

                # ======== Train Generator ========
                optimizer_G.zero_grad()
                
                # Get target in dual format for L1 loss
                dual_target_x = create_dual_graph_feature_matrix(target_m)
                
                # L1 loss in dual space - scale down to balance with adversarial loss
                l1_weight = 10.0  # Adjust this weight as needed
                loss_G_L1 = criterion_L1(dual_pred_x, dual_target_x) * l1_weight
                
                # Adversarial loss - get fresh discriminator output (no detach)
                fake_output = discriminator(fake_adj)
                # Target 1.0 for generator
                loss_G_adv = criterion_BCE(fake_output, torch.ones_like(fake_output))
                
                # Calculate generator fool rate
                g_fool_acc = ((fake_output > 0.5).float().mean()).item()
                
                # Total generator loss - weighted sum
                loss_G = loss_G_L1 + loss_G_adv
                
                # Periodically print detailed loss breakdown
                if batch_counter % 50 == 0:
                    print(f"  Batch {batch_counter}: G_L1={loss_G_L1.item():.4f}, G_adv={loss_G_adv.item():.4f}, "
                          f"D_real={loss_D_real.item():.4f}, D_fake={loss_D_fake.item():.4f}")
                    print(f"  Accuracies: D_real={d_real_acc:.4f}, D_fake={d_fake_acc:.4f}, G_fool={g_fool_acc:.4f}")
                
                loss_G.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

                # Update counters and accumulators
                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()
                epoch_d_real_acc += d_real_acc
                epoch_d_fake_acc += d_fake_acc
                epoch_g_fool_acc += g_fool_acc
                batch_counter += 1

            # Calculate epoch averages
            n_batches = len(source_data_train)
            epoch_loss_G /= n_batches
            epoch_loss_D /= n_batches
            epoch_d_real_acc /= n_batches
            epoch_d_fake_acc /= n_batches
            epoch_g_fool_acc /= n_batches
            
            # Store losses and metrics
            train_losses_G.append(epoch_loss_G)
            train_losses_D.append(epoch_loss_D)
            d_real_accuracies.append(epoch_d_real_acc)
            d_fake_accuracies.append(epoch_d_fake_acc)
            g_fool_accuracies.append(epoch_g_fool_acc)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{config.experiment.n_epochs}")
            print(f"  Generator Loss: {epoch_loss_G:.4f}, Discriminator Loss: {epoch_loss_D:.4f}")
            print(f"  Discriminator Real Accuracy: {epoch_d_real_acc:.4f}, Fake Accuracy: {epoch_d_fake_acc:.4f}")
            print(f"  Generator Fool Rate: {epoch_g_fool_acc:.4f}")

            # Sample visualization - show what the generator is producing
            if epoch % 5 == 0:
                with torch.no_grad():
                    idx = np.random.randint(0, len(source_data_val))
                    sample_source_g = source_data_val[idx]['pyg'].to(device)
                    sample_target_m = target_data_val[idx]['mat'].to(device)
                    
                    sample_dual_pred = generator(sample_source_g, sample_target_m)
                    sample_fake_adj = revert_dual(sample_dual_pred, n_target_nodes)
                    
                    print_tensor_stats(sample_fake_adj, f"Epoch {epoch} Generated Adj")
                    # Optional: save or visualize matrices
            
            if config.experiment.log_val_loss:
                # Validation
                val_loss = evaluate(config, generator, source_data_val, target_data_val, criterion_L1)
                val_losses.append(val_loss)
                print(f"  Validation Loss: {val_loss:.4f}")

        # Save and plot losses
        torch.save(generator.state_dict(), f"{res_dir}/generator.pth")
        torch.save(discriminator.state_dict(), f"{res_dir}/discriminator.pth")
        np.save(f'{res_dir}/train_losses_G.npy', np.array(train_losses_G))
        np.save(f'{res_dir}/train_losses_D.npy', np.array(train_losses_D))
        np.save(f'{res_dir}/val_losses.npy', np.array(val_losses))
        np.save(f'{res_dir}/d_real_acc.npy', np.array(d_real_accuracies))
        np.save(f'{res_dir}/d_fake_acc.npy', np.array(d_fake_accuracies))
        np.save(f'{res_dir}/g_fool_acc.npy', np.array(g_fool_accuracies))
        
        # Plot losses and accuracy metrics
        plot_losses(train_losses_G, 'train_losses_G', res_dir)
        plot_losses(train_losses_D, 'train_losses_D', res_dir)
        plot_losses(val_losses, 'val', res_dir)
        plot_losses(d_real_accuracies, 'd_real_acc', res_dir)
        plot_losses(d_fake_accuracies, 'd_fake_acc', res_dir)
        plot_losses(g_fool_accuracies, 'g_fool_acc', res_dir)

    return {
        'generator': generator,
        'discriminator': discriminator,
    }

def evaluate(config, generator, source_data_val, target_data_val, criterion):
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
            
            # Get predictions in dual space
            dual_pred_x = generator(source_g, target_m)
            
            # Convert target to dual space
            dual_target_x = create_dual_graph_feature_matrix(target_m)
            
            # Calculate loss in dual space
            loss = criterion(dual_pred_x, dual_target_x)
            total_loss += loss.item()
    
    generator.train()
    return total_loss / len(source_data_val)

# Function for visualizing adjacency matrices
def visualize_adjacency_matrix(adj_matrix, title, save_path=None):
    """
    Visualize an adjacency matrix.
    
    Args:
        adj_matrix (torch.Tensor): The adjacency matrix to visualize
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(8, 8))
    
    # Convert to numpy if it's a tensor
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    plt.imshow(adj_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()