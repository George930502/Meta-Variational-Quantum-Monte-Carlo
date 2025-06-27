import matplotlib.pyplot as plt
import numpy as np

def plot_vmc_comparison(results, config):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        'Pre-trained VMC': 'blue',
        'Random Init VMC': 'black',
    }

    for label, data in results.items():
        steps = np.arange(len(data))
        ax.plot(steps, data, label=label, color=colors.get(label, 'gray'), linewidth=2)

    ax.set_xlabel("VMC Iteration", fontsize=14)
    ax.set_ylabel("Energy", fontsize=14)
    title = f"VMC Convergence for {config['system']['name']}"
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig("vmc_comparison.png", dpi=300)
    plt.show()