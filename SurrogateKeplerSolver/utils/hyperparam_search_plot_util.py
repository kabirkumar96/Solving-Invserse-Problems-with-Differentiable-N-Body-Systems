from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_DT_hyperparam_search(hyperparam_search_results, model_name, sub_name, savefig = False):
    # Sort and extract data
    sorted_results = sorted(hyperparam_search_results, key=lambda x: x['max_depth'])
    max_depths = [result['max_depth'] for result in sorted_results]
    val_losses = [result['val_loss'] for result in sorted_results]
    
    # Find minimum
    min_loss_idx = np.argmin(val_losses)
    min_loss = val_losses[min_loss_idx]
    min_loss_depth = max_depths[min_loss_idx]
    
    plt.figure(figsize=(12, 6))
    
    # Plot main line and points
    plt.semilogy(max_depths, val_losses, 'o-', label='Loss', color='#2ecc71', linewidth=2)
    
    # Plot minimum point with different color
    plt.plot(min_loss_depth, min_loss, 'o', color='red', markersize=8)
    
    # Styling
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title('Decision Tree Performance vs Max Depth', pad=20)
    plt.xlabel('Max Depth')
    plt.ylabel('Loss (log scale)')
    
    plt.annotate(f'{min_loss:.2e}',
                (min_loss_depth, min_loss),
                textcoords="offset points",
                xytext=(-5,10),
                ha='center',
                fontsize=8)
    
    plt.legend(loc='upper right')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.1)
    plt.margins(x=0.02)
    plt.tight_layout()

    if savefig:
        plt.savefig(
            f"figures/hyperparam_search_figures/{model_name}/hyperparam_search{sub_name}")
    
    return plt.show()

def plot_RF_hyperparam_search(hyperparam_search_results, model_name, sub_name, savefig = False):
    # Extract unique values
    n_estimators = sorted(set(result['n_estimators'] for result in hyperparam_search_results))
    max_depth = sorted(set(result['max_depth'] for result in hyperparam_search_results))
    
    # Create a 2D grid to store validation losses
    val_losses = np.zeros((len(n_estimators), len(max_depth)))
    val_losses[:] = np.nan  # Fill with NaNs initially
    
    # Populate the grid
    for result in hyperparam_search_results:
        n_estimators_index = n_estimators.index(result['n_estimators'])
        max_depth_index = max_depth.index(result['max_depth'])
        val_losses[n_estimators_index, max_depth_index] = result['val_loss']
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    
    # Create annotation array with empty strings
    annot = np.full_like(val_losses, '', dtype='U10')
    
    # Find minimum loss position and value
    min_idx = np.unravel_index(np.nanargmin(val_losses), val_losses.shape)
    min_loss = val_losses[min_idx]
    
    # Only annotate the minimum value
    annot[min_idx] = f'{min_loss:0.2e}'
    
    sns.heatmap(val_losses,
                annot=annot,
                fmt='',
                cmap='YlGnBu',
                xticklabels=max_depth,
                yticklabels=n_estimators,
                norm=LogNorm(),  # Add logarithmic normalization
                cbar_kws={'label': 'Loss (log scale)'})
    
    plt.title('RF Hyperparameter Search: Loss Heatmap')
    plt.xlabel('Max Depth')
    plt.ylabel('N Estimators')
    plt.tight_layout()

    if savefig:
        plt.savefig(
            f"figures/hyperparam_search_figures/{model_name}/hyperparam_search{sub_name}")
        
    plt.show()
    

def plot_ANN_hyperparam_search(hyperparam_search_results, model_name, sub_name, savefig = False):
    # Extract unique values
    num_layers = sorted(set(result['num_layers'] for result in hyperparam_search_results))
    layer_widths = sorted(set(result['layer_widths'] for result in hyperparam_search_results))
    
    # Create a 2D grid to store validation losses
    val_losses = np.zeros((len(num_layers), len(layer_widths)))
    val_losses[:] = np.nan  # Fill with NaNs initially
    
    # Populate the grid
    for result in hyperparam_search_results:
        layer_index = num_layers.index(result['num_layers'])
        width_index = layer_widths.index(result['layer_widths'])
        val_losses[layer_index, width_index] = result['val_loss']
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    
    # Create annotation array with empty strings
    annot = np.full_like(val_losses, '', dtype='U10')
    
    # Find minimum loss position and value
    min_idx = np.unravel_index(np.nanargmin(val_losses), val_losses.shape)
    min_loss = val_losses[min_idx]
    
    # Only annotate the minimum value
    annot[min_idx] = f'{min_loss:0.2e}'

    sns.heatmap(val_losses,
                annot=annot,
                fmt='',
                cmap='YlGnBu',
                xticklabels=layer_widths,
                yticklabels=num_layers,
                norm=LogNorm(),  # Add logarithmic normalization
                cbar_kws={'label': 'Loss (log scale)'})
    
    plt.title('ANN Hyperparameter Search: Loss Heatmap')
    plt.xlabel('Layer Width')
    plt.ylabel('Number of Layers')
    plt.tight_layout()

    if savefig:
        plt.savefig(
            f"figures/hyperparam_search_figures/{model_name}/hyperparam_search{sub_name}")

    plt.show()