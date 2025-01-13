import numpy as np
from . import TensorCode_util as tc


def perturb_masses(m, perturbation_scale=0.01):
    """
    Perturb masses with a small random variation using uniform sampling.

    Parameters:
    -----------
    m : numpy.ndarray
        Original masses array
    perturbation_scale : float, optional
        Scale of mass perturbation (default: 1%)

    Returns:
    --------
    numpy.ndarray
        Perturbed masses array
    """
    # Generate random perturbations from a uniform distribution
    perturbations = np.random.uniform(
        low=1.0 - perturbation_scale,  # Lower bound of uniform distribution
        high=1.0 + perturbation_scale,  # Upper bound of uniform distribution
        size=m.shape
    )
    # Apply perturbations to masses
    perturbed_masses = m * perturbations
    return perturbed_masses


def perturb_positions(r, perturbation_scale=0.01):
    """
    Perturb positions with a small random variation using uniform sampling.

    Parameters:
    -----------
    r : numpy.ndarray
        Original positions array
    perturbation_scale : float, optional
        Scale of position perturbation (default: 1%)

    Returns:
    --------
    numpy.ndarray
        Perturbed positions array
    """
    # Generate random perturbations from a uniform distribution
    perturbations = np.random.uniform(
        low=-perturbation_scale,  # Negative lower bound
        high=perturbation_scale,  # Positive upper bound
        size=r.shape
    )
    # Apply perturbations to positions
    perturbed_positions = r + perturbations
    return perturbed_positions


def perturb_velocities(v, perturbation_scale=0.01):
    """
    Perturb velocities with a small random variation using uniform sampling.

    Parameters:
    -----------
    v : numpy.ndarray
        Original velocities array
    perturbation_scale : float, optional
        Scale of velocity perturbation (default: 1%)

    Returns:
    --------
    numpy.ndarray
        Perturbed velocities array
    """
    # Generate random perturbations from a uniform distribution
    perturbations = np.random.uniform(
        low=-perturbation_scale,  # Negative lower bound
        high=perturbation_scale,  # Positive upper bound
        size=v.shape
    )
    # Apply perturbations to velocities
    perturbed_velocities = v + perturbations
    return perturbed_velocities


def generate_perturbed_ensemble(
    tau,  # Time step
    n,  # Number of bodies
    m,  # Masses
    r,  # Starting positions
    v,  # Starting velocities
    k=10,  # Number of ensemble members
    num_steps=2500,  # Simulation steps per ensemble member
    mass_perturbation_scale=0.01,  # Mass perturbation scale
    position_perturbation_scale=0.01,  # Position perturbation scale
    velocity_perturbation_scale=0.01,  # Velocity perturbation scale
):
    """
    Generate an ensemble of Kepler simulations with perturbed parameters.

    Parameters:
    -----------
    k : int, optional
        Number of ensemble members (default: 10)
    tau : float
        Time step
    n : int
        Number of bodies
    m : numpy.ndarray
        Initial masses
    r : numpy.ndarray
        Initial positions
    v : numpy.ndarray
        Initial velocities
    num_steps : int, optional
        Number of simulation steps per ensemble member (default: 2500)
    mass_perturbation_scale : float, optional
        Scale of mass perturbation (default: 1%)
    position_perturbation_scale : float, optional
        Scale of position perturbation (default: 1%)
    velocity_perturbation_scale : float, optional
        Scale of velocity perturbation (default: 1%)

    Returns:
    --------
    dict: Containing ensemble data for each run
    """
    # Initialize ensemble storage
    ensemble = {
        'r_t_ensemble': [],  # Position time series for each run
        'v_t_ensemble': [],  # Velocity time series for each run
        'kepler_data_ensemble': [],  # Kepler data for each run
        'perturbed_masses_ensemble': [],  # Perturbed masses for each run
        'perturbed_positions_ensemble': [],  # Perturbed positions for each run
        'perturbed_velocities_ensemble': []  # Perturbed velocities for each run
    }

    # Run ensemble simulation
    for ensemble_idx in range(k):
        # Initialize perturbed parameters
        m_perturbed = m
        r_perturbed = r
        v_perturbed = v

        # Perturb parameters for non-initial ensemble members
        if k > 0:
            m_perturbed = perturb_masses(m, mass_perturbation_scale)
            r_perturbed = perturb_positions(r, position_perturbation_scale)
            v_perturbed = perturb_velocities(v, velocity_perturbation_scale)

        # Initialize tracking arrays
        r_t = [r_perturbed]
        v_t = [v_perturbed]
        all_kepler_data = None

        # Run simulation with perturbed parameters
        for step in range(num_steps):
            # Combine positions and velocities
            rv = np.column_stack([r_perturbed, v_perturbed])

            # Do simulation step with perturbed parameters
            rv1, kepler_data = tc.do_step_wrapper_normal(
                tau, n, m_perturbed, rv, gen_kepler_data=True)

            # Split results
            r1, v1 = np.split(rv1, indices_or_sections=2, axis=-1)

            # Accumulate Kepler data
            if all_kepler_data is None:
                all_kepler_data = kepler_data
            else:
                all_kepler_data = np.vstack((all_kepler_data, kepler_data))

            # Update positions and velocities
            r_perturbed, v_perturbed = r1, v1
            r_t.append(r_perturbed)
            v_t.append(v_perturbed)

        # Convert to numpy arrays
        r_t = np.array(r_t)
        v_t = np.array(v_t)

        # Store results for this ensemble member
        ensemble['r_t_ensemble'].append(r_t)
        ensemble['v_t_ensemble'].append(v_t)
        ensemble['kepler_data_ensemble'].append(all_kepler_data)
        ensemble['perturbed_masses_ensemble'].append(m_perturbed)
        ensemble['perturbed_positions_ensemble'].append(r_perturbed)
        ensemble['perturbed_velocities_ensemble'].append(v_perturbed)

    return ensemble


# def simulate_d
