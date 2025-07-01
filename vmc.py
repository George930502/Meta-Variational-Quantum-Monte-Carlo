import torch
import torch.nn as nn
import math

# def mcmc_sampler(model, n_samples, num_spins, mcmc_params):
#     device = next(model.parameters()).device
#     burn_in_sweeps = mcmc_params.get('burn_in_sweeps', 500)
#     decorrelation_sweeps = mcmc_params.get('decorrelation_sweeps', 100)

#     current_spins = (torch.randint(0, 2, (1, num_spins), device=device) * 2 - 1).float()

#     with torch.no_grad():
#         log_psi_current = model(current_spins)

#         # --- Burn-in ---
#         for _ in range(burn_in_sweeps * num_spins):
#             flip_idx = torch.randint(0, num_spins, ()).item()
#             spins_proposed = current_spins.clone()
#             spins_proposed[0, flip_idx] *= -1
#             log_psi_proposed = model(spins_proposed)
#             accept = torch.rand(()) < torch.exp(2 * (log_psi_proposed - log_psi_current)).clamp(max=1.0)
#             if accept:
#                 current_spins = spins_proposed
#                 log_psi_current = log_psi_proposed

#         # --- Sampling ---
#         samples = torch.empty((n_samples, num_spins), device=device)
#         for i in range(n_samples):
#             for _ in range(decorrelation_sweeps * num_spins):
#                 flip_idx = torch.randint(0, num_spins, ()).item()
#                 spins_proposed = current_spins.clone()
#                 spins_proposed[0, flip_idx] *= -1
#                 log_psi_proposed = model(spins_proposed)
#                 accept = torch.rand(()) < torch.exp(2 * (log_psi_proposed - log_psi_current)).clamp(max=1.0)
#                 if accept:
#                     current_spins = spins_proposed
#                     log_psi_current = log_psi_proposed
#             samples[i] = current_spins

#     return samples


def mcmc_sampler_batch(model, n_samples, num_spins, mcmc_params, batch_size):
    device = next(model.parameters()).device
    burn_in_sweeps = mcmc_params.get('burn_in_sweeps')
    decorrelation_sweeps = mcmc_params.get('decorrelation_sweeps')

    current_spins = (torch.randint(0, 2, (batch_size, num_spins), device=device) * 2 - 1).float()
    
    with torch.no_grad():
        log_psi_current = model(current_spins)

        # --- Burn-in ---
        # print(f"Burn-in phase: {burn_in_sweeps} sweeps with batch size {batch_size}.")
        for _ in range(burn_in_sweeps * num_spins):
            flip_idx = torch.randint(0, num_spins, (batch_size,), device=device) 
            spins_proposed = current_spins.clone()
            spins_proposed[torch.arange(batch_size), flip_idx] *= -1

            log_psi_proposed = model(spins_proposed)
            delta = 2 * (log_psi_proposed - log_psi_current)
            accept = (torch.rand(batch_size, device=device) < torch.exp(delta.clamp(max=0.0)))
            current_spins[accept] = spins_proposed[accept]
            log_psi_current[accept] = log_psi_proposed[accept]

        # --- Sampling with decorrelation ---
        # print(f"Sampling phase: {n_samples} samples with batch size {batch_size}, decorrelation sweeps {decorrelation_sweeps}.")
        samples = []
        n_rounds = math.ceil(n_samples / batch_size)

        for _ in range(n_rounds):
            for _ in range(decorrelation_sweeps * num_spins):
                flip_idx = torch.randint(0, num_spins, (batch_size,), device=device)
                spins_proposed = current_spins.clone()
                spins_proposed[torch.arange(batch_size), flip_idx] *= -1

                log_psi_proposed = model(spins_proposed)
                delta = 2 * (log_psi_proposed - log_psi_current)
                accept = (torch.rand(batch_size, device=device) < torch.exp(delta.clamp(max=0.0)))
                current_spins[accept] = spins_proposed[accept]
                log_psi_current[accept] = log_psi_proposed[accept]
            
            samples.append(current_spins.clone())

        # print(f"Generated {len(samples) * batch_size} samples, returning {n_samples} samples.")
        samples = torch.cat(samples, dim=0)
        return samples[:n_samples]


def get_local_energies(hamiltonian, model, spins):
    device = spins.device
    unique_spins, inverse_indices = torch.unique(spins, dim=0, return_inverse=True)
    
    with torch.no_grad():
        log_psi_unique = model(unique_spins)
    
    num_spins = spins.shape[1]
    diag_energy = torch.zeros(spins.shape[0], device=device)
    off_diag_energy = torch.zeros(spins.shape[0], device=device)

    if 'J' in hamiltonian:
        J = torch.tensor(hamiltonian['J'], dtype=torch.float32, device=device)
        diag_energy -= torch.einsum('bi,ij,bj->b', spins, torch.triu(J, diagonal=1), spins)
    
    if 'g' in hamiltonian:
        g = torch.tensor(hamiltonian['g'], dtype=torch.float32, device=device)
        diag_energy -= torch.einsum('bi,i->b', spins, g)

    if 'h' in hamiltonian:
        h = torch.tensor(hamiltonian['h'], dtype=torch.float32, device=device)

        # Vectorized spin flipping
        expanded = unique_spins.repeat_interleave(num_spins, dim=0)
        flip_idx = torch.arange(num_spins, device=device).repeat(unique_spins.size(0))
        expanded[torch.arange(len(expanded)), flip_idx] *= -1

        with torch.no_grad():
            log_psi_flipped = model(expanded)
        
        log_psi_repeated = log_psi_unique.repeat_interleave(num_spins)
        psi_ratios = torch.exp(log_psi_flipped - log_psi_repeated).view(unique_spins.size(0), num_spins).T
        psi_batch = psi_ratios[:, inverse_indices]
        off_diag_energy = -torch.einsum('i,ib->b', h, psi_batch)

    return (diag_energy + off_diag_energy).detach()


def vmc_loss(model, hamiltonian, cfg):
    vmc_cfg = cfg['vmc']
    num_spins = getattr(model, 'num_visible', getattr(model, 'num_spins', None))
    
    # spins = mcmc_sampler(model, vmc_cfg['n_samples'], num_spins, vmc_cfg['mcmc'])
    spins = mcmc_sampler_batch(model, vmc_cfg['n_samples'], num_spins, vmc_cfg['mcmc'], vmc_cfg['batch_size'])

    spins = spins.detach()  
    log_psi = model(spins)
    local_energies = get_local_energies(hamiltonian, model, spins)

    e_mean = local_energies.mean()
    loss = torch.mean((log_psi - log_psi.mean()) * (local_energies - e_mean))

    return loss, e_mean
