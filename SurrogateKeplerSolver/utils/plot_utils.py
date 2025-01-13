import matplotlib.pyplot as plt
import numpy as np
import shap

def plot_SHAP_values(explainer, data_input, filepath, sub_name = "", title = 'SHAP Values Across All Outputs', savefig = False):

    shap_values = explainer.shap_values(data_input)

    fig, ax = plt.subplots(2, 3, figsize=(20, 15))
    input_features = ['mij', 'r[0]', 'r[1]', 'r[2]', 'v[0]', 'v[1]', 'v[2]']
    output_features = ['r_out[0]', 'r_out[1]', 'r_out[2]', 'v_out[0]', 'v_out[1]', 'v_out[2]']

    for i in range(6):
        plt.subplot(2, 3, i+1)
        shap.summary_plot(shap_values[..., i], data_input, 
                            plot_type="bar",
                            show=False,
                            feature_names = input_features)
        plt.xticks(fontsize=8)
        plt.xlabel(f"{output_features[i]}", fontsize = 10)

    plt.suptitle(title, fontsize=12)  # Add main titleplt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(f"figures/trained_models_figures/{filepath}/SHAP_scores{sub_name}")
    return fig, ax


def plot_trajectory_comparison(actual_data, predicted_data, n, filepath, starting_point, num_steps, figsize=(12, 3), markevery=100, savefig=False, sub_name=""):
    """
    Create side-by-side comparison plots of true and predicted trajectories.

    Parameters:
    -----------
    actual_data : numpy.ndarray
        True trajectory data with shape (timesteps, n_planets, coordinates)
    predicted_data : numpy.ndarray
        Predicted trajectory data with shape (timesteps, n_planets, coordinates)
    n : int
        Number of planets
    figsize : tuple, optional
        Figure size in inches (width, height), default (12, 3)
    markevery : int, optional
        Interval for marking points on trajectories, default 100

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    (ax1, ax2) : tuple
        Tuple containing the two axis objects
    """
    # Create a single figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # First plot (true trajectories)
    for i_planet in range(n):
        ax1.plot(*actual_data[starting_point:num_steps, i_planet, :2].T, '.-', markevery=markevery,
                 label=f'Planet {i_planet+1}')
    ax1.set_title('True Trajectories')
    ax1.legend()

    # Second plot (predicted trajectories)
    for i_planet in range(n):
        ax2.plot(*predicted_data[starting_point:num_steps, i_planet, :2].T, '.-', markevery=markevery,
                 label=f'Planet {i_planet+1}')
    ax2.set_title('Predicted Trajectories')
    ax2.legend()

    # Get the limits across both plots
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    # Set the same limits for both plots
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    if savefig:
        plt.savefig(
            f"figures/trained_models_figures/{filepath}/trajectory_comparison{sub_name}")

    return fig, (ax1, ax2)


def plot_pointwise_mse(actual_data, predicted_data, filepath, num_steps=None, savefig=False, sub_name="", error_threshold=1.0):
    """
    Plot pointwise Mean Squared Error (MSE) for each body across steps.
    Parameters:
    -----------
    actual_data : numpy array
        Actual trajectory data of shape [num_steps, num_bodies, 3]
    predicted_data : numpy array
        Predicted trajectory data of shape [num_steps, num_bodies, 3]
    num_steps : int, optional
        Number of steps to plot (defaults to all steps)
    error_threshold : float, optional
        Threshold for marking first high error point (default: 1.0)
    Returns:
    --------
    fig : matplotlib Figure
        Figure containing the MSE plots
    axes : array of matplotlib Axes
        Axes for further customization if needed
    """
    # Default to full trajectory if num_steps not specified
    if num_steps is None:
        num_steps = actual_data.shape[0]
    num_bodies = actual_data.shape[1]

    # Create figure with subplots
    fig, axes = plt.subplots(num_bodies, 1, figsize=(12, 4*num_bodies),
                             sharex=True)

    # Ensure axes is always an array, even for single body
    if num_bodies == 1:
        axes = [axes]

    # Compute and plot pointwise MSE for each body
    for body_idx in range(num_bodies):
        # Compute MSE along each dimension
        mse = np.mean((actual_data[:num_steps, body_idx] -
                      predicted_data[:num_steps, body_idx])**2, axis=1)

        # Plot MSE vs steps
        # print(mse[mse == 0])
        axes[body_idx].plot(range(num_steps), mse,
                            label=f'Body {body_idx+1} MSE')

        # Find first point where error exceeds threshold
        first_error_idx = None
        for t in range(num_steps):
            if mse[t] > error_threshold:
                first_error_idx = t
                break

        # Plot vertical dotted line and step number at first error point if found
        if first_error_idx is not None:
            axes[body_idx].axvline(x=first_error_idx, color='red',
                                   linestyle=':', linewidth=2,
                                   label=f'First Error > {error_threshold}')

            # Add text annotation with step number
            # Get y-axis limits to position the text
            ymin, ymax = axes[body_idx].get_ylim()
            text_y = ymax * 0.95  # Position text near top of plot

            axes[body_idx].text(first_error_idx + num_steps*0.01, text_y,
                                f'Step {first_error_idx}',
                                color='red', verticalalignment='top')

        axes[body_idx].set_ylabel('Pointwise MSE')
        axes[body_idx].set_title(f'Pointwise MSE for Body {body_idx+1}')
        axes[body_idx].legend()
        axes[body_idx].grid(True, alpha=0.3)

    # Set common x-axis label
    fig.text(0.5, 0.04, 'Number of Steps', ha='center')

    if savefig:
        plt.savefig(
            f"figures/trained_models_figures/{filepath}/pointwise_planet_MSE{sub_name}")

    # plt.tight_layout()
    return fig, axes


def plot_distance_vs_mse(actual_data, predicted_data, filepath, num_steps=None, savefig=False, sub_name=""):
    """
    Plot the relationship between inter-body distances and prediction MSE.
    For 3 bodies: Creates 3D plots with distances between bodies on X/Y and MSE on Z
    For 2 bodies: Creates 2D plots with distance between bodies on X and MSE on Y

    Parameters:
    actual_data: numpy array of shape [num_steps, num_bodies, 3]
    predicted_data: numpy array of shape [num_steps, num_bodies, 3]
    num_steps: optional, number of steps to plot (defaults to all steps)
    """
    if num_steps is None:
        num_steps = actual_data.shape[0]

    num_bodies = actual_data.shape[1]
    figsize = (15, 5)

    # Create a figure with subplots for each body
    if num_bodies == 2:
        fig, axes = plt.subplots(1, num_bodies, figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        axes = []
        for i in range(num_bodies):
            ax = fig.add_subplot(1, num_bodies, i+1, projection='3d')
            axes.append(ax)

    # Make axes iterable if there's only one subplot
    if num_bodies == 2:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

    for body_idx in range(num_bodies):
        # Calculate MSE for current body
        mse = np.mean((actual_data[:num_steps, body_idx] -
                      predicted_data[:num_steps, body_idx])**2, axis=1)

        if num_bodies == 2:
            # For 2 bodies: plot distance vs MSE
            # Calculate distance between bodies
            distances = np.sqrt(
                np.sum((actual_data[:num_steps, 0] - actual_data[:num_steps, 1])**2, axis=1))

            # Create 2D plot
            axes[body_idx].scatter(distances, mse, alpha=0.5, c='blue')
            axes[body_idx].set_xlabel('Distance between bodies')
            axes[body_idx].set_ylabel('MSE')
            axes[body_idx].set_title(f'Body {body_idx + 1}')
            axes[body_idx].grid(True, alpha=0.3)

        else:
            # For 3 bodies: calculate distances between current body and others
            distances = {}
            other_bodies = [i for i in range(num_bodies) if i != body_idx]

            for other_idx in other_bodies:
                distances[other_idx] = np.sqrt(np.sum(
                    (actual_data[:num_steps, body_idx] -
                     actual_data[:num_steps, other_idx])**2,
                    axis=1
                ))

            # Create 3D scatter plot
            scatter = axes[body_idx].scatter(
                distances[other_bodies[0]],
                distances[other_bodies[1]],
                mse,
                c=mse,  # Color points by MSE value
                cmap='viridis',
                alpha=0.6
            )

            # Add labels and title
            axes[body_idx].set_xlabel(
                f'Distance to Body {other_bodies[0] + 1}')
            axes[body_idx].set_ylabel(
                f'Distance to Body {other_bodies[1] + 1}')
            axes[body_idx].set_zlabel('MSE')
            axes[body_idx].set_title(f'Body {body_idx + 1}')

            # Add colorbar
            plt.colorbar(scatter, ax=axes[body_idx], label='MSE')

    if savefig:
        plt.savefig(
            f"figures/trained_models_figures/{filepath}/dist_vs_MSE{sub_name}")
    # plt.tight_layout()
    return fig, axes


def plot_planetary_trajectories(actual_data, predicted_data, filepath, starting_point=0, num_steps=None, savefig=False, sub_name="", error_threshold=1.0, markevery=1):
    """
    Plot actual vs predicted trajectories for each planetary body in separate subplots.
    Parameters:
    actual_data: numpy array of shape [num_steps, num_planets, 3]
    predicted_data: numpy array of shape [num_steps, num_planets, 3]
    num_steps: optional, number of steps to plot (defaults to all steps)
    error_threshold: threshold for marking first high error point (default: 1.0)
    """
    if num_steps is None:
        num_steps = actual_data.shape[0]

    num_planets = actual_data.shape[1]

    # Create a figure with num_planets subplots arranged vertically
    fig, axes = plt.subplots(num_planets, 1, figsize=(10, 4*num_planets))

    # Add spacing between subplots
    plt.subplots_adjust(hspace=0.3)  # Increase this value to add more space

    if num_planets == 1:
        axes = [axes]

    # Colors for actual and predicted trajectories
    actual_color = '#1f77b4'  # blue
    predicted_color = '#ff7f0e'  # orange
    error_color = 'red'  # color for first high error point

    for i in range(num_planets):
        ax = axes[i]
        # Plot actual trajectory
        ax.plot(actual_data[starting_point:(starting_point+num_steps), i, 0],
                actual_data[starting_point:(starting_point+num_steps), i, 1],
                'o-', color=actual_color, markersize=2,
                label='Actual', alpha=0.7, markevery=markevery)

        # Plot predicted trajectory
        ax.plot(predicted_data[starting_point:(starting_point+num_steps), i, 0],
                predicted_data[starting_point:(
                    starting_point+num_steps), i, 1],
                'o-', color=predicted_color, markersize=2,
                label='Predicted', alpha=0.7, markevery=markevery)

        # Find first point where error exceeds threshold
        first_error_point = None
        for t in range(starting_point, starting_point + num_steps):
            # Calculate MSE for current position
            mse = np.mean(
                (actual_data[t, i, :2] - predicted_data[t, i, :2])**2)
            # If MSE exceeds threshold for the first time, store the point and break
            if mse > error_threshold:
                first_error_point = t
                break

        # Plot the first error point if found
        if first_error_point is not None:
            ax.plot(actual_data[first_error_point, i, 0],
                    actual_data[first_error_point, i, 1],
                    'o', color=error_color, markersize=8, alpha=0.7,
                    label='First Error > Threshold')

        # Add start and end markers
        ax.plot(actual_data[starting_point, i, 0], actual_data[starting_point, i, 1],
                '*', color='green', label='Start', markersize=14)
        ax.plot(actual_data[starting_point+num_steps-1, i, 0], actual_data[starting_point+num_steps-1, i, 1],
                '*', color='red', label='End', markersize=14)

        # Customize subplot
        # Added pad parameter to increase space between title and plot
        ax.set_title(f'Planet {i+1} Trajectory')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')  # Equal scaling on both axes

    if savefig:
            plt.savefig(
                f"figures/trained_models_figures/{filepath}/planetary_trajectories{sub_name}")
        # plt.tight_layout()
    return fig, axes
