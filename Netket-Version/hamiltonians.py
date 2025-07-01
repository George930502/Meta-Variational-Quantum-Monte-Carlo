import netket as nk
import numpy as np
import jax.numpy as jnp


def create_hilbert_space(num_spins):
    """Create the Hilbert space for spin-1/2 systems."""
    return nk.hilbert.Spin(s=1/2, N=num_spins)


def maxcut_hamiltonian(hilbert_space, A_matrix):
    """Create MaxCut Hamiltonian from adjacency matrix."""
    # MaxCut Hamiltonian: H = sum_{i,j} A_{ij} (1 - σ_z^i σ_z^j) / 4
    # This is equivalent to: H = sum_{i,j} A_{ij}/4 - sum_{i,j} A_{ij} σ_z^i σ_z^j / 4
    
    operators = []
    
    # Add the ZZ interaction terms
    for i in range(hilbert_space.size):
        for j in range(i+1, hilbert_space.size):
            if A_matrix[i, j] != 0:
                # -A_{ij}/4 * σ_z^i σ_z^j term (negative because we want to minimize cuts)
                operators.append([-A_matrix[i, j]/4, nk.operator.spin.sigmaz(hilbert_space, i) @ 
                                 nk.operator.spin.sigmaz(hilbert_space, j)])
    
    # Add constant term: sum_{i,j} A_{ij}/4
    # constant = np.sum(np.triu(A_matrix, 1)) / 4
    # if len(operators) > 0:
    #     operators.append([constant, nk.operator.spin.identity(hilbert_space)])
    
    # if len(operators) == 0:
    #     return nk.operator.spin.identity(hilbert_space) * 0
    
    return sum(coeff * op for coeff, op in operators)


def sk_hamiltonian(hilbert_space, J_matrix, h_field=None, g_field=None):
    """Create Sherrington-Kirkpatrick Hamiltonian."""
    operators = []
    
    # ZZ interactions
    for i in range(hilbert_space.size):
        for j in range(i+1, hilbert_space.size):
            if J_matrix[i, j] != 0:
                operators.append([-J_matrix[i, j], nk.operator.spin.sigmaz(hilbert_space, i) @ 
                                 nk.operator.spin.sigmaz(hilbert_space, j)])
    
    # Transverse field (X terms)
    if h_field is not None:
        for i in range(hilbert_space.size):
            if h_field[i] != 0:
                operators.append([-h_field[i], nk.operator.spin.sigmax(hilbert_space, i)])
    
    # Longitudinal field (Z terms)
    if g_field is not None:
        for i in range(hilbert_space.size):
            if g_field[i] != 0:
                operators.append([-g_field[i], nk.operator.spin.sigmaz(hilbert_space, i)])
    
    if len(operators) == 0:
        return nk.operator.identity(hilbert_space) * 0
    
    return sum(coeff * op for coeff, op in operators)


def tfim_hamiltonian(hilbert_space, J_matrix, h_field):
    """Create Transverse Field Ising Model Hamiltonian."""
    operators = []
    
    # ZZ interactions from coupling matrix
    for i in range(hilbert_space.size):
        for j in range(i+1, hilbert_space.size):
            if J_matrix[i, j] != 0:
                operators.append([-J_matrix[i, j], nk.operator.spin.sigmaz(hilbert_space, i) @ 
                                 nk.operator.spin.sigmaz(hilbert_space, j)])
    
    # Transverse field
    for i in range(hilbert_space.size):
        if h_field[i] != 0:
            operators.append([-h_field[i], nk.operator.spin.sigmax(hilbert_space, i)])
    
    if len(operators) == 0:
        return nk.operator.identity(hilbert_space) * 0
    
    return sum(coeff * op for coeff, op in operators)


def hamiltonian_from_dict(hilbert_space, hamiltonian_dict):
    """Convert a hamiltonian dictionary to a NetKet operator."""
    ham_type = hamiltonian_dict['type']
    
    if ham_type == 'diagonal':
        # MaxCut case
        J_matrix = hamiltonian_dict['J']
        # Convert Laplacian-based formulation back to adjacency matrix
        # L = D - A, so A = D - L, but we need to be careful about the factor of 0.25
        A_matrix = -4 * J_matrix  # Reverse the 0.25 factor
        np.fill_diagonal(A_matrix, 0)  # Ensure diagonal is zero
        return maxcut_hamiltonian(hilbert_space, A_matrix)
    
    elif ham_type == 'non_diagonal':
        # TFIM case
        J_matrix = hamiltonian_dict['J']
        h_field = hamiltonian_dict['h']
        return tfim_hamiltonian(hilbert_space, J_matrix, h_field)
    
    elif ham_type == 'non_diagonal_with_longitudinal':
        # SK case
        J_matrix = hamiltonian_dict['J']
        h_field = hamiltonian_dict.get('h', None)
        g_field = hamiltonian_dict.get('g', None)
        return sk_hamiltonian(hilbert_space, J_matrix, h_field, g_field)
    
    else:
        raise ValueError(f"Unknown Hamiltonian type: {ham_type}")


def get_hamiltonian_and_task_generator(config):
    """Create task generator that returns NetKet Hamiltonians."""
    problem_type = config['experiment']['problem_type']
    num_spins = config['problem_params']['num_spins']
    sigma = config['experiment']['sigma']
    
    # Create the Hilbert space once
    hilbert_space = create_hilbert_space(num_spins)
    
    if problem_type == 'MaxCut':
        B = np.random.uniform(0, 1, size=(num_spins, num_spins))
        A_base = (B + B.T) / 2
        np.fill_diagonal(A_base, 0)
        
        def task_generator():
            noise = np.random.normal(loc=0, scale=sigma, size=(num_spins, num_spins))
            delta_A = (noise + noise.T) / 2
            A_task = np.clip(np.round(A_base + delta_A), 0, 1)
            np.fill_diagonal(A_task, 0)

            D_task = np.diag(np.sum(A_task, axis=1))
            L_task = D_task - A_task
            J_matrix = 0.25 * L_task
            
            hamiltonian_dict = {'type': 'diagonal', 'J': J_matrix}
            hamiltonian = hamiltonian_from_dict(hilbert_space, hamiltonian_dict)
            return hamiltonian, hilbert_space
            
        return task_generator

    elif problem_type == 'SK':
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
            
            hamiltonian_dict = {'type': 'non_diagonal_with_longitudinal', 
                              'J': task_g_ij, 'h': task_h_i, 'g': task_g_i}
            hamiltonian = hamiltonian_from_dict(hilbert_space, hamiltonian_dict)
            return hamiltonian, hilbert_space
            
        return task_generator

    elif problem_type in ['TFIM1D', 'TFIM2D']:
        h_base_val = config['problem_params']['transverse_field']
        
        if problem_type == 'TFIM1D':
            base_g = np.full(num_spins, 1.0)
            base_h = np.full(num_spins, h_base_val)
            
            def task_generator():
                task_g = base_g + np.random.normal(0, sigma, num_spins)
                task_h = np.clip(base_h + np.random.normal(0, sigma, num_spins), 0, None)
                J_matrix = np.zeros((num_spins, num_spins))
                for i in range(num_spins):
                    J_matrix[i, (i + 1) % num_spins] = task_g[i]
                J_matrix = (J_matrix + J_matrix.T) / 2
                
                hamiltonian_dict = {'type': 'non_diagonal', 'J': J_matrix, 'h': task_h}
                hamiltonian = hamiltonian_from_dict(hilbert_space, hamiltonian_dict)
                return hamiltonian, hilbert_space
                
            return task_generator
            
        else:  # TFIM2D
            L = int(np.sqrt(num_spins))
            if L * L != num_spins: 
                raise ValueError("TFIM2D requires a square number of spins.")
            base_g_h = np.full((L, L), 1.0)
            base_g_v = np.full((L, L), 1.0)
            base_h = np.full((L, L), h_base_val)
            
            def task_generator():
                task_g_h = base_g_h + np.random.normal(0, sigma, (L, L))
                task_g_v = base_g_v + np.random.normal(0, sigma, (L, L))
                task_h_flat = np.clip((base_h + np.random.normal(0, sigma, (L, L))).flatten(), 0, None)
                J_matrix = np.zeros((num_spins, num_spins))
                for i in range(L):
                    for j in range(L):
                        idx = i * L + j
                        J_matrix[idx, i * L + (j + 1) % L] = task_g_h[i, j]
                        J_matrix[idx, ((i + 1) % L) * L + j] = task_g_v[i, j]
                J_matrix = (J_matrix + J_matrix.T) / 2
                
                hamiltonian_dict = {'type': 'non_diagonal', 'J': J_matrix, 'h': task_h_flat}
                hamiltonian = hamiltonian_from_dict(hilbert_space, hamiltonian_dict)
                return hamiltonian, hilbert_space
                
            return task_generator
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


if __name__ == "__main__":
    # Example usage
    config = {
        'experiment': {
            'problem_type': 'MaxCut',
            'sigma': 0.1
        },
        'problem_params': {
            'num_spins': 10,
            'transverse_field': 1.0
        }
    }
    
    task_gen = get_hamiltonian_and_task_generator(config)
    hamiltonian, hilbert_space = task_gen()
    print(hamiltonian)
    print(f"Created Hamiltonian for {hilbert_space.size} spins")
    print(f"Hamiltonian type: {type(hamiltonian)}") 