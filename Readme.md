# Meta Variational Quantum Monte Carlo

This project is a PyTorch-based reproduction of the experiments from the paper 
[Meta Variational Quantum Monte Carlo](https://link.springer.com/article/10.1007/s42484-022-00094-w) (Zhao et al.).  It implements the Model-Agnostic Meta-Learning (MAML) algorithm to find good initial parameters for Variational Monte Carlo (VMC) simulations of various quantum spin systems.

## Project Structure

```tree 
├── README.md             # 專案說明與執行指南
├── config.yaml           # 集中管理所有實驗配置
├── requirements.txt      # 依賴套件
├── main.py               # 主執行腳本， orchestrates everything
├── models.py             # RBM 和 CNN 神經網路架構
├── hamiltonians.py       # 定義和生成四種物理問題的哈密頓量
├── vmc.py                # VMC 核心邏輯 (包含非對角哈密頓量的處理)
├── maml.py               # MAML/foMAML 訓練與評估邏輯
└── utils.py              # 繪圖及其他輔助函數
```

- `README.md`: This file, providing an overview and instructions.
- `config.yaml`: The central configuration file to manage all experiments. **This is the main file you will edit to choose an experiment.**
- `requirements.txt`: A list of required Python packages for the project.
- `main.py`: The main entry point to run experiments. It orchestrates the training and evaluation process.
- `models.py`: Contains the definitions for the neural network architectures (RBM and CNN) used as the variational wavefunction.
- `hamiltonians.py`: Defines the physics. It generates the Hamiltonians and task distributions for Max-Cut, Sherrington-Kirkpatrick (SK), and 1D/2D Transverse Field Ising Models (TFIM).
- `vmc.py`: The core Variational Monte Carlo logic, including the calculation of local energies for both diagonal and non-diagonal Hamiltonians.
- `maml.py`: Implements the MAML and foMAML training and evaluation loops.
- `utils.py`: Helper functions, primarily for plotting the final results (Energy vs. Iteration).


## Setup

1.  Clone the repository or save all the provided files into a single directory.
2.  Create a virtual environment (recommended) and install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run an Experiment

1.  **Edit `config.yaml`**: Open the `config.yaml` file and choose the experiment you want to run. You need to set `problem_type` and `model_type` to match the figures in the paper.

    **Example: To run the 2D Transverse Field Ising Model experiment with a CNN model:**
    ```yaml
    experiment:
      problem_type: 'TFIM2D' # Options: 'MaxCut', 'SK', 'TFIM1D', 'TFIM2D'
      model_type: 'CNN'     # Options: 'RBM', 'CNN'
      sigma: 0.5            # Task diversity (standard deviation)
    ```

2.  **Run the main script** from your terminal:
    ```bash
    python main.py
    ```

3.  The script will:
    - Read the configuration from `config.yaml`.
    - Run the MAML training to get optimized initial parameters.
    - Run the foMAML training similarly.
    - Set up a baseline with random initial parameters.
    - Evaluate all three initializations by fine-tuning them on a set of test tasks.
    - Generate a plot named `training_curves.png` comparing the energy convergence.

## Expected Output

The script will produce a plot showing the average energy as a function of fine-tuning iterations for MAML-SGD, foMAML-SGD, and Rand Init-SGD. This plot is saved as `training_curves.png` in the project directory, ready for comparison with the paper's results.