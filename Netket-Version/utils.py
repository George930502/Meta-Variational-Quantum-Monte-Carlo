import matplotlib.pyplot as plt
import numpy as np


def plot_curves(results, config, selected_algorithms=None):
    """
    Plot energy convergence curves for selected algorithms.
    
    Args:
        results: Dictionary with algorithm names as keys and energy curves as values
        config: Configuration dictionary
        selected_algorithms: List of algorithm names to plot (subset of results.keys()).
                             If None, plot all.
    """

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'MAML-SGD': 'blue',
        'foMAML-SGD': 'red', 
        'Rand Init-SGD': 'black',
    }

    # Determine which algorithms to plot
    if selected_algorithms is None:
        selected_algorithms = list(results.keys())

    for label in selected_algorithms:
        if label not in results:
            print(f"Warning: '{label}' not found in results. Skipping.")
            continue
        
        energy_curves = results[label]
        mean_energies = np.mean(energy_curves, axis=1)
        std_energies = np.std(energy_curves, axis=1)
        
        steps = np.arange(len(mean_energies))
        color = colors.get(label, 'gray')
        
        # Plot mean with error bars
        ax.plot(steps, mean_energies, label=label, color=color, linewidth=2)
        ax.fill_between(steps, 
                        mean_energies - std_energies, 
                        mean_energies + std_energies, 
                        color=color, alpha=0.2)

    ax.set_xlabel("Fine-tuning Iteration", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    
    problem_type = config['experiment']['problem_type']
    model_type = config['experiment']['model_type']
    num_spins = config['problem_params']['num_spins']
    sigma = config['experiment']['sigma']
    
    title = f"Meta-Learning VMC: {problem_type} ({model_type}, N={num_spins}, σ={sigma})"
    ax.set_title(title, fontsize=16)
    
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_curves_ALL.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_energy_statistics(results, config):
    """Print final energy statistics for each algorithm."""
    print("\n" + "="*50)
    print("FINAL ENERGY STATISTICS")
    print("="*50)
    
    for label, energy_curves in results.items():
        final_energies = energy_curves[-1, :]  # Last step energies for all tasks
        mean_final = np.mean(final_energies)
        std_final = np.std(final_energies)
        min_final = np.min(final_energies)
        max_final = np.max(final_energies)
        
        print(f"\n{label}:")
        print(f"  Mean final energy: {mean_final:.6f} ± {std_final:.6f}")
        print(f"  Min final energy:  {min_final:.6f}")
        print(f"  Max final energy:  {max_final:.6f}")


def save_results(results, config, filename="results.npz"):
    """Save results to a numpy archive file."""
    np.savez(filename, 
             results=results,
             config=config)
    print(f"Results saved to {filename}")


def load_results(filename="results.npz"):
    """Load results from a numpy archive file."""
    data = np.load(filename, allow_pickle=True)
    return data['results'].item(), data['config'].item()


def compute_improvement_metrics(results):
    """Compute improvement metrics comparing MAML/foMAML to random baseline."""
    if 'Rand Init-SGD' not in results:
        print("Warning: No random baseline found for comparison")
        return
    
    baseline = results['Rand Init-SGD']
    baseline_final = np.mean(baseline[-1, :])
    
    print("\n" + "="*50)
    print("IMPROVEMENT METRICS")
    print("="*50)
    
    for label, energy_curves in results.items():
        if label == 'Rand Init-SGD':
            continue
            
        final_energies = np.mean(energy_curves[-1, :])
        improvement = baseline_final - final_energies
        relative_improvement = (improvement / abs(baseline_final)) * 100
        
        print(f"\n{label} vs Random Baseline:")
        print(f"  Absolute improvement: {improvement:.6f}")
        print(f"  Relative improvement: {relative_improvement:.2f}%")


def plot_energy_distribution(results, config):
    """Plot histogram of final energies for each algorithm."""
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (label, energy_curves) in enumerate(results.items()):
        final_energies = energy_curves[-1, :]
        
        axes[idx].hist(final_energies, bins=10, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"{label}\nFinal Energy Distribution", fontsize=12)
        axes[idx].set_xlabel("Energy", fontsize=10)
        axes[idx].set_ylabel("Frequency", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("energy_distributions.png", dpi=300, bbox_inches='tight')
    plt.show() 