import matplotlib.pyplot as plt
import numpy as np

def plot_curves(results, config):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'MAML-SGD': 'blue',
        'foMAML-SGD': 'red',
        'Rand Init-SGD': 'black',
    }

    for label, data in results.items():
        mean_curve = np.mean(data, axis=1)
        steps = np.arange(len(mean_curve))
        
        ax.plot(steps, mean_curve, label=label, color=colors.get(label, 'gray'), linewidth=2)

    ax.set_xlabel("Number of Iterations", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    title = (f"Training Curves for {config['experiment']['problem_type']} with {config['experiment']['model_type']}\n"
             f"(σ = {config['experiment']['sigma']})")
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()