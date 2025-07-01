import jax
import jax.numpy as jnp
import netket as nk
import optax
from models import create_netket_machine


def create_vmc_state(config, hilbert_space):
    """Create a NetKet variational state for VMC."""
    return create_netket_machine(config, hilbert_space)


def vmc_loss_and_energy(vstate, hamiltonian):
    """
    Compute VMC loss and energy using NetKet's built-in functionality.
    Returns (loss, energy) where loss is the variance-based loss for gradient computation.
    """
    # Compute local energies
    local_energies = vstate.local_estimators(hamiltonian)
    
    # Compute mean energy
    energy_mean = jnp.mean(local_energies)
    
    # Compute log probabilities
    log_psi = vstate.log_value(vstate.samples)
    
    # VMC loss: covariance-based estimator
    # This is the standard VMC gradient estimator
    centered_log_psi = log_psi - jnp.mean(log_psi)
    centered_energies = local_energies - energy_mean
    
    loss = 2 * jnp.mean(centered_log_psi * centered_energies)
    
    return loss, energy_mean


def single_vmc_step(vstate, hamiltonian, optimizer_state, optimizer):
    """Perform a single VMC optimization step."""
    def loss_fn(params):
        vstate_copy = vstate
        vstate_copy.parameters = params
        return vmc_loss_and_energy(vstate_copy, hamiltonian)[0]
    
    # Compute gradients
    loss_val, grads = jax.value_and_grad(loss_fn)(vstate.parameters)
    
    # Apply optimizer update
    updates, optimizer_state = optimizer.update(grads, optimizer_state, vstate.parameters)
    new_params = optax.apply_updates(vstate.parameters, updates)
    
    # Update variational state
    new_vstate = vstate
    new_vstate.parameters = new_params
    
    # Compute energy with updated parameters
    _, energy = vmc_loss_and_energy(new_vstate, hamiltonian)
    
    return new_vstate, optimizer_state, loss_val, energy


def run_vmc_optimization(config, hamiltonian, hilbert_space, initial_params=None, n_steps=None):
    """
    Run VMC optimization using NetKet components.
    
    Args:
        config: Configuration dictionary
        hamiltonian: NetKet Hamiltonian operator
        hilbert_space: NetKet Hilbert space
        initial_params: Optional initial parameters
        n_steps: Number of optimization steps (defaults to config value)
    
    Returns:
        (final_vstate, energy_history)
    """
    # Create variational state
    vstate = create_vmc_state(config, hilbert_space)
    
    # Set initial parameters if provided
    if initial_params is not None:
        vstate.parameters = initial_params
    
    # Create optimizer
    learning_rate = config['maml']['inner_lr']
    optimizer = optax.sgd(learning_rate)
    optimizer_state = optimizer.init(vstate.parameters)
    
    # Number of steps
    if n_steps is None:
        n_steps = config['evaluation']['finetune_steps']
    
    energy_history = []
    
    # Optimization loop
    for step in range(n_steps):
        # Sample new configurations
        vstate.sample()
        
        # Perform optimization step
        vstate, optimizer_state, loss_val, energy = single_vmc_step(
            vstate, hamiltonian, optimizer_state, optimizer
        )
        
        energy_history.append(float(energy))
    
    return vstate, jnp.array(energy_history)


def evaluate_energy(vstate, hamiltonian):
    """Evaluate the energy of a given variational state."""
    # Sample configurations
    vstate.sample()
    
    # Compute local energies
    local_energies = vstate.local_estimators(hamiltonian)
    
    # Return mean energy
    return jnp.mean(local_energies)


class VMCLoss:
    """Wrapper class for VMC loss computation compatible with meta-learning."""
    
    def __init__(self, config):
        self.config = config
    
    def __call__(self, params, vstate_template, hamiltonian):
        """
        Compute VMC loss for given parameters.
        
        Args:
            params: Model parameters
            vstate_template: Template variational state (for structure)
            hamiltonian: NetKet Hamiltonian
        
        Returns:
            (loss, energy)
        """
        # Create variational state with given parameters
        vstate = create_vmc_state(self.config, vstate_template.hilbert)
        vstate.parameters = params
        
        # Sample configurations
        vstate.sample()
        
        # Compute loss and energy
        return vmc_loss_and_energy(vstate, hamiltonian) 
    