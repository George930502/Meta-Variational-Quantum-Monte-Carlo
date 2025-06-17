import torch

# get_local_energies function remains the same as before

def get_local_energies(hamiltonian, model, spins):
    device = spins.device
    log_psi_s = model(spins)

    if hamiltonian['type'] == 'diagonal':
        J = torch.from_numpy(hamiltonian['J']).float().to(device)
        J_upper = torch.triu(J, diagonal=1)
        diag_energy = -torch.einsum('bi,ij,bj->b', spins.float(), J_upper, spins.float())
        return diag_energy.detach()

    elif hamiltonian['type'] == 'non_diagonal':
        J = torch.from_numpy(hamiltonian['J']).float().to(device)
        h = torch.from_numpy(hamiltonian['h']).float().to(device)
        num_spins = spins.shape[1]
        
        diag_energy = -torch.einsum('bi,ij,bj->b', spins.float(), torch.triu(J, diagonal=1), spins.float())
        
        off_diag_energy = torch.zeros_like(diag_energy)
        for i in range(num_spins):
            spins_flipped = spins.clone()
            spins_flipped[:, i] *= -1
            log_psi_s_flipped = model(spins_flipped)
            psi_ratio = torch.exp(log_psi_s_flipped - log_psi_s)
            off_diag_energy -= h[i] * psi_ratio
        return (diag_energy + off_diag_energy).detach()
    
    elif hamiltonian['type'] == 'non_diagonal_with_longitudinal':
        J = torch.from_numpy(hamiltonian['J']).float().to(device) # g_ij
        h = torch.from_numpy(hamiltonian['h']).float().to(device) # h_i
        g = torch.from_numpy(hamiltonian['g']).float().to(device) # g_i
        num_spins = spins.shape[1]

        diag_energy_zz = -torch.einsum('bi,ij,bj->b', spins.float(), torch.triu(J, diagonal=1), spins.float())
        diag_energy_z = -torch.einsum('bi,i->b', spins.float(), g)
        diag_energy = diag_energy_zz + diag_energy_z

        off_diag_energy = torch.zeros_like(diag_energy)
        for i in range(num_spins):
            spins_flipped = spins.clone()
            spins_flipped[:, i] *= -1
            log_psi_s_flipped = model(spins_flipped)
            psi_ratio = torch.exp(log_psi_s_flipped - log_psi_s)
            off_diag_energy -= h[i] * psi_ratio
        return (diag_energy + off_diag_energy).detach()

    else:
        raise ValueError("Unknown Hamiltonian type")


def vmc_loss(model, hamiltonian, n_samples):
    # --- FIX: Get num_spins reliably from the model attribute ---
    if hasattr(model, 'num_visible'):
        num_spins = model.num_visible
    elif hasattr(model, 'num_spins'):
        num_spins = model.num_spins
    else:
        # Fallback with a more informative error
        raise AttributeError("Model must have a 'num_visible' or 'num_spins' attribute.")
        
    device = next(model.parameters()).device
    
    # Simple MCMC sampling
    spins = (torch.randint(0, 2, (n_samples, num_spins), device=device) * 2 - 1).float()
    
    log_psi = model(spins)
    local_energies = get_local_energies(hamiltonian, model, spins)
    
    e_mean = local_energies.mean()
    loss = torch.mean((log_psi - log_psi.mean()) * (local_energies - e_mean))
    
    return loss, e_mean