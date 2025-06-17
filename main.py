import yaml
import torch
import numpy as np

from models import get_model
from hamiltonians import get_hamiltonian_and_task_generator
from maml import run_maml_or_fomaml, evaluate
from utils import plot_curves

def main():
    # --- 1. Load Configuration ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- 2. Setup problem and model ---
    task_generator = get_hamiltonian_and_task_generator(config)
    
    # We need a way to create new models of the same class for evaluation
    model_builder = lambda: get_model(config)

    results = {}

    # --- 3. Run Experiments ---
    
    # MAML
    print("\n" + "="*20 + " Running MAML-SGD " + "="*20)
    config['training'] = {'algorithm': 'MAML'}
    maml_model_init = model_builder()
    maml_params = run_maml_or_fomaml(config, maml_model_init, task_generator)
    results['MAML-SGD'] = evaluate(config, maml_params, model_builder(), task_generator)

    # foMAML
    print("\n" + "="*20 + " Running foMAML-SGD " + "="*20)
    config['training'] = {'algorithm': 'foMAML'}
    fomaml_model_init = model_builder()
    fomaml_params = run_maml_or_fomaml(config, fomaml_model_init, task_generator)
    results['foMAML-SGD'] = evaluate(config, fomaml_params, model_builder(), task_generator)

    # Baseline: Random Initialization
    print("\n" + "="*20 + " Running Rand Init-SGD " + "="*20)
    rand_model = model_builder()
    # We just use the initial (random) parameters of the model
    rand_init_params = rand_model.state_dict()
    results['Rand Init-SGD'] = evaluate(config, rand_init_params, model_builder(), task_generator)
    
    # --- 4. Plot Results ---
    print("\nPlotting results...")
    plot_curves(results, config)


if __name__ == '__main__':
    main()