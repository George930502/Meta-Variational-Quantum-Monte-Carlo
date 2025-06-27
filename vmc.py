import torch
import torch.nn as nn

def mcmc_sampler(model, n_samples, num_spins, mcmc_params):
    device = next(model.parameters()).device
    burn_in_sweeps = mcmc_params.get('burn_in_sweeps', 100)
    decorrelation_sweeps = mcmc_params.get('decorrelation_sweeps', 10)
    
    current_spins = (torch.randint(0, 2, (1, num_spins), device=device) * 2 - 1).float()
    
    with torch.no_grad():
        log_psi_current = model(current_spins)

        for _ in range(burn_in_sweeps * num_spins):
            flip_idx = torch.randint(0, num_spins, (1,)).item()
            spins_proposed = current_spins.clone()
            spins_proposed[0, flip_idx] *= -1
            log_psi_proposed = model(spins_proposed)
            acceptance_prob = torch.exp(2 * (log_psi_proposed - log_psi_current)).clamp(max=1.0)
            if torch.rand(1).item() < acceptance_prob:
                current_spins = spins_proposed
                log_psi_current = log_psi_proposed
        
        samples_list = []
        for _ in range(n_samples):
            for _ in range(decorrelation_sweeps * num_spins):
                flip_idx = torch.randint(0, num_spins, (1,)).item()
                spins_proposed = current_spins.clone()
                spins_proposed[0, flip_idx] *= -1
                log_psi_proposed = model(spins_proposed)
                acceptance_prob = torch.exp(2 * (log_psi_proposed - log_psi_current)).clamp(max=1.0)
                if torch.rand(1).item() < acceptance_prob:
                    current_spins = spins_proposed
                    log_psi_current = log_psi_proposed
            samples_list.append(current_spins.clone())
            
    return torch.cat(samples_list, dim=0)

def get_local_energies(hamiltonian, model, spins):
    device = spins.device
    unique_spins, inverse_indices = torch.unique(spins, dim=0, return_inverse=True)
    log_psi_unique = model(unique_spins)
    log_psi_s = log_psi_unique[inverse_indices]
    
    num_spins = spins.shape[1]
    diag_energy = torch.zeros(spins.shape[0], device=device)
    off_diag_energy = torch.zeros(spins.shape[0], device=device)

    if 'J' in hamiltonian:
        J = torch.from_numpy(hamiltonian['J']).float().to(device)
        diag_energy -= torch.einsum('bi,ij,bj->b', spins.float(), torch.triu(J, diagonal=1), spins.float())
    if 'g' in hamiltonian:
        g = torch.from_numpy(hamiltonian['g']).float().to(device)
        diag_energy -= torch.einsum('bi,i->b', spins.float(), g)
    if 'h' in hamiltonian:
        h = torch.from_numpy(hamiltonian['h']).float().to(device)
        all_flipped_spins = unique_spins.repeat(num_spins, 1)
        flip_indices = torch.arange(num_spins, device=device).repeat_interleave(unique_spins.shape[0])
        all_flipped_spins[torch.arange(all_flipped_spins.shape[0]), flip_indices] *= -1
        log_psi_flipped_unique_all = model(all_flipped_spins)
        psi_ratios_unique_all = torch.exp(log_psi_flipped_unique_all - log_psi_unique.repeat(num_spins))
        psi_ratios_unique = psi_ratios_unique_all.view(num_spins, unique_spins.shape[0])
        psi_ratios_batch = psi_ratios_unique[:, inverse_indices]
        off_diag_energy = -torch.einsum('i,ib->b', h, psi_ratios_batch)

    return (diag_energy + off_diag_energy).detach()

def vmc_loss(model, hamiltonian, cfg):
    vmc_cfg = cfg['vmc']
    if hasattr(model, 'num_visible'):
        num_spins = model.num_visible
    else:
        num_spins = model.num_spins
        
    spins = mcmc_sampler(model, vmc_cfg['n_samples'], num_spins, vmc_cfg['mcmc'])
    
    log_psi = model(spins)
    local_energies = get_local_energies(hamiltonian, model, spins)
    
    e_mean = local_energies.mean()
    loss = torch.mean((log_psi - log_psi.mean()) * (local_energies - e_mean))
    
    return loss, e_mean