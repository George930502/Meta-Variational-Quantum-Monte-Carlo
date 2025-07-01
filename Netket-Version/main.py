import yaml
import jax
import numpy as np
import jax.numpy as jnp
from jax import random

from hamiltonians import get_hamiltonian_and_task_generator
from maml import run_maml_or_fomaml, evaluate, create_random_baseline
from utils import plot_curves


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set random seeds for reproducibility
    np.random.seed(config['seed'])
    key = random.PRNGKey(config['seed'])
    
    # Create task generator
    task_generator = get_hamiltonian_and_task_generator(config)
    
    results = {}

    # --- MAML ---
    print("\n" + "="*20 + " Running MAML-SGD " + "="*20)
    config['training'] = {'algorithm': 'MAML'}
    maml_params = run_maml_or_fomaml(config, task_generator)
    results['MAML-SGD'] = evaluate(config, maml_params, task_generator)

    # --- foMAML ---
    print("\n" + "="*20 + " Running foMAML-SGD " + "="*20)
    config['training'] = {'algorithm': 'foMAML'}
    fomaml_params = run_maml_or_fomaml(config, task_generator)
    results['foMAML-SGD'] = evaluate(config, fomaml_params, task_generator)

    # --- Baseline: Random Initialization ---
    print("\n" + "="*20 + " Running Rand Init-SGD " + "="*20)
    rand_params = create_random_baseline(config, task_generator)
    results['Rand Init-SGD'] = evaluate(config, rand_params, task_generator)
    
    # --- Plot Results ---
    print("\nPlotting results...")
    plot_curves(results, config, selected_algorithms=['MAML-SGD', 'foMAML-SGD', 'Rand Init-SGD'])
    
    print("\nExperiment completed! Results saved as 'training_curves.png'")


if __name__ == '__main__':
    main() 