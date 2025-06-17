import numpy as np

def get_hamiltonian_and_task_generator(config):
    problem_type = config['experiment']['problem_type']
    num_spins = config['problem_params']['num_spins']
    sigma = config['experiment']['sigma']

    if problem_type == 'MaxCut':
        # Correct implementation based on paper
        B = np.random.randint(0, 2, size=(num_spins, num_spins))
        A_base = np.triu(B, 1) + np.triu(B, 1).T

        def task_generator():
            noise = np.random.normal(loc=0, scale=sigma, size=(num_spins, num_spins))
            delta_A = (noise + noise.T) / 2
            A_task = np.round(np.clip(A_base + delta_A, 0, 1))
            np.fill_diagonal(A_task, 0)
            D_task = np.diag(np.sum(A_task, axis=1))
            L_task = D_task - A_task
            J_matrix = 0.25 * L_task
            return {'type': 'diagonal', 'J': J_matrix}
        return task_generator

    elif problem_type == 'SK':
        # Correct implementation based on paper
        base_g_ij = np.random.uniform(-1, 1, (num_spins, num_spins))
        base_g_ij = (base_g_ij + base_g_ij.T) / 2
        np.fill_diagonal(base_g_ij, 0)
        base_g_i = np.random.uniform(-1, 1, num_spins)
        base_h_i = np.random.uniform(0, 1, num_spins)

        def task_generator():
            noise_g_ij = np.random.normal(0, sigma, (num_spins, num_spins))
            task_g_ij = base_g_ij + (noise_g_ij + noise_g_ij.T) / 2
            np.fill_diagonal(task_g_ij, 0)
            task_g_i = base_g_i + np.random.normal(0, sigma, num_spins)
            task_h_i = np.clip(base_h_i + np.random.normal(0, sigma, num_spins), 0, None)
            return {'type': 'non_diagonal_with_longitudinal', 'J': task_g_ij, 'h': task_h_i, 'g': task_g_i}
        return task_generator

    elif problem_type in ['TFIM1D', 'TFIM2D']:
        # ===================================================================
        # CORRECTED TFIM IMPLEMENTATION
        # ===================================================================
        h_base_val = config['problem_params']['transverse_field']

        if problem_type == 'TFIM1D':
            # Base parameters: g_i for each bond, h_i for each site
            base_g = np.full(num_spins, 1.0)
            base_h = np.full(num_spins, h_base_val)

            def task_generator():
                # Perturb parameters
                task_g = base_g + np.random.normal(0, sigma, num_spins)
                task_h = np.clip(base_h + np.random.normal(0, sigma, num_spins), 0, None)

                # Construct the J matrix from bond-dependent g_i values
                J_matrix = np.zeros((num_spins, num_spins))
                for i in range(num_spins):
                    J_matrix[i, (i + 1) % num_spins] = task_g[i]
                J_matrix = (J_matrix + J_matrix.T) / 2 # Symmetrize

                return {'type': 'non_diagonal', 'J': J_matrix, 'h': task_h}
            return task_generator

        else: # TFIM2D
            L = int(np.sqrt(num_spins))
            if L * L != num_spins:
                raise ValueError("TFIM2D requires a square number of spins.")

            # Base parameters: separate for horizontal/vertical bonds and sites
            base_g_h = np.full((L, L), 1.0) # Horizontal couplings
            base_g_v = np.full((L, L), 1.0) # Vertical couplings
            base_h = np.full((L, L), h_base_val) # Site fields

            def task_generator():
                # Perturb all parameters
                task_g_h = base_g_h + np.random.normal(0, sigma, (L, L))
                task_g_v = base_g_v + np.random.normal(0, sigma, (L, L))
                task_h_flat = np.clip((base_h + np.random.normal(0, sigma, (L, L))).flatten(), 0, None)

                # Construct J matrix from bond-dependent g values
                J_matrix = np.zeros((num_spins, num_spins))
                for i in range(L):
                    for j in range(L):
                        idx = i * L + j
                        # Horizontal bond
                        right_idx = i * L + (j + 1) % L
                        J_matrix[idx, right_idx] = task_g_h[i, j]
                        # Vertical bond
                        down_idx = ((i + 1) % L) * L + j
                        J_matrix[idx, down_idx] = task_g_v[i, j]
                J_matrix = (J_matrix + J_matrix.T) / 2 # Symmetrize

                return {'type': 'non_diagonal', 'J': J_matrix, 'h': task_h_flat}
            return task_generator
        # ===================================================================
        # END OF TFIM CORRECTION
        # ===================================================================

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
    

if __name__ == "__main__":
    # Example configuration
    config = {
        'experiment': {
            'problem_type': 'MaxCut',
            'sigma': 0.1
        },
        'problem_params': {
            'num_spins': 49
        }
    }
    
    task_gen = get_hamiltonian_and_task_generator(config)
    task = task_gen()
    print(task) 