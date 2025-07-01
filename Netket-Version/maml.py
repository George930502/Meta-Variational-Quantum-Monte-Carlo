import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import numpy as np

from vmc import create_vmc_state, VMCLoss
from models import create_netket_machine


def inner_update(params, vstate_template, hamiltonian, loss_fn, learning_rate):
    """Perform a single inner update step."""
    def single_loss(p):
        return loss_fn(p, vstate_template, hamiltonian)[0]
    
    loss_val, grads = jax.value_and_grad(single_loss)(params)
    
    # SGD update
    updated_params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    
    return updated_params, loss_val


def meta_loss_fn(params, vstate_template, hamiltonian, loss_fn, inner_lr, inner_steps=1):
    """Compute meta-loss after inner updates."""
    current_params = params
    
    # Perform inner updates
    for _ in range(inner_steps):
        current_params, _ = inner_update(current_params, vstate_template, hamiltonian, 
                                       loss_fn, inner_lr)
    
    # Compute loss with updated parameters
    meta_loss, _ = loss_fn(current_params, vstate_template, hamiltonian)
    return meta_loss


def fomaml_meta_loss_fn(params, vstate_template, hamiltonian, loss_fn, inner_lr, inner_steps=1):
    """Compute foMAML meta-loss (first-order approximation)."""
    # For foMAML, we use stop_gradient to avoid computing higher-order derivatives
    current_params = params
    
    # Perform inner updates with stop_gradient
    for _ in range(inner_steps):
        current_params, _ = inner_update(current_params, vstate_template, hamiltonian, 
                                       loss_fn, inner_lr)
        current_params = jax.lax.stop_gradient(current_params)
    
    # Compute loss with updated parameters
    meta_loss, _ = loss_fn(current_params, vstate_template, hamiltonian)
    return meta_loss


def run_maml_or_fomaml(config, hamiltonian_task_generator):
    """
    Run MAML or foMAML training using NetKet components.
    
    Args:
        config: Configuration dictionary
        hamiltonian_task_generator: Function that generates (hamiltonian, hilbert_space) pairs
    
    Returns:
        optimized_params: The meta-learned parameters
    """
    # Get algorithm type
    is_fomaml = (config['training']['algorithm'] == 'foMAML')
    
    # Create a template task to initialize the model
    sample_hamiltonian, hilbert_space = hamiltonian_task_generator()
    
    # Create template variational state
    vstate_template = create_netket_machine(config, hilbert_space)
    # Get initial parameters from the already initialized variational state
    initial_params = vstate_template.parameters
    
    # Create meta-optimizer
    meta_optimizer = optax.sgd(config['maml']['meta_lr'])
    meta_opt_state = meta_optimizer.init(initial_params)
    
    # Create loss function
    loss_fn = VMCLoss(config)
    
    # Meta-training parameters
    meta_epochs = config['maml']['meta_epochs']
    meta_batch_size = config['maml']['meta_batch_size']
    inner_lr = config['maml']['inner_lr']
    inner_steps = config['maml']['inner_steps']
    
    # Choose the appropriate meta-loss function
    if is_fomaml:
        meta_loss_func = fomaml_meta_loss_fn
        print(f"Starting meta-training for foMAML...")
    else:
        meta_loss_func = meta_loss_fn
        print(f"Starting meta-training for MAML...")
    
    current_params = initial_params
    
    # Meta-training loop
    for meta_epoch in range(meta_epochs):
        epoch_meta_loss = 0.0
        
        # Compute meta-gradients over a batch of tasks
        def batch_meta_loss(params):
            batch_loss = 0.0
            for _ in range(meta_batch_size):
                task_hamiltonian, task_hilbert = hamiltonian_task_generator()
                
                # Create a new variational state for this specific task
                task_vstate = create_netket_machine(config, task_hilbert)
                # Set the parameters to our current meta-parameters
                task_vstate.parameters = params
                # Sample new configurations for this task
                task_vstate.sample()
                
                task_loss = meta_loss_func(params, task_vstate, task_hamiltonian, 
                                         loss_fn, inner_lr, inner_steps)
                batch_loss += task_loss
            
            return batch_loss / meta_batch_size
        
        # Compute meta-gradients
        meta_loss_val, meta_grads = jax.value_and_grad(batch_meta_loss)(current_params)
        
        # Update meta-parameters
        updates, meta_opt_state = meta_optimizer.update(meta_grads, meta_opt_state, current_params)
        current_params = optax.apply_updates(current_params, updates)
        
        epoch_meta_loss = float(meta_loss_val)

        print(f"Meta-Epoch [{meta_epoch+1}/{meta_epochs}] Meta-Loss: {epoch_meta_loss:.6f}")
    
    print("Meta-training finished.")
    return current_params


def evaluate(config, initial_params, hamiltonian_task_generator):
    """
    Evaluate the meta-learned parameters on test tasks.
    
    Args:
        config: Configuration dictionary
        initial_params: Initial parameters (either meta-learned or random)
        hamiltonian_task_generator: Function that generates test tasks
    
    Returns:
        energy_curves: Array of shape (finetune_steps, num_test_tasks)
    """
    num_test_tasks = config['evaluation']['num_test_tasks']
    finetune_steps = config['evaluation']['finetune_steps']
    inner_lr = config['maml']['inner_lr']
    
    energy_curves = np.zeros((finetune_steps, num_test_tasks))
    
    # Get template structures
    sample_hamiltonian, hilbert_space = hamiltonian_task_generator()
    vstate_template = create_netket_machine(config, hilbert_space)
    loss_fn = VMCLoss(config)
    
    for task_idx in tqdm(range(num_test_tasks), desc="Evaluating"):
        # Generate test task
        test_hamiltonian, test_hilbert = hamiltonian_task_generator()
        
        # Initialize with meta-learned parameters
        current_params = initial_params
        
        # Create task-specific variational state
        task_vstate = create_netket_machine(config, test_hilbert)
        task_vstate.parameters = current_params
        
        # Fine-tuning loop
        for step in range(finetune_steps):
            # Sample configurations
            task_vstate.sample()
            
            # Perform one optimization step
            current_params, loss_val = inner_update(current_params, task_vstate, 
                                                   test_hamiltonian, loss_fn, inner_lr)
            
            # Update variational state with new parameters
            task_vstate.parameters = current_params
            
            # Compute and store energy
            _, energy = loss_fn(current_params, task_vstate, test_hamiltonian)
            energy_curves[step, task_idx] = float(energy)
    
    return energy_curves


def create_random_baseline(config, hamiltonian_task_generator):
    """Create random baseline parameters for comparison."""
    # Get template structures
    sample_hamiltonian, hilbert_space = hamiltonian_task_generator()
    vstate_template = create_netket_machine(config, hilbert_space)
    
    # Get parameters from the already initialized variational state
    # (create_netket_machine already initializes with random parameters)
    random_params = vstate_template.parameters
    
    return random_params 