# maml.py (Corrected)
import torch
from torch.optim import SGD
import higher
from tqdm import tqdm
import numpy as np

from vmc import vmc_loss

# run_maml_or_fomaml function remains the same

def run_maml_or_fomaml(cfg, model, task_generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    meta_optimizer = SGD(model.parameters(), lr=cfg['maml']['meta_lr'])
    is_fomaml = (cfg['training']['algorithm'] == 'foMAML')

    print(f"Starting meta-training for {cfg['training']['algorithm']}...")
    for meta_epoch in range(cfg['maml']['meta_epochs']):
        meta_optimizer.zero_grad()
        meta_grad_accumulator = [torch.zeros_like(p, device=device) for p in model.parameters()]
        
        for _ in range(cfg['maml']['meta_batch_size']):
            task_H = task_generator()
            inner_opt = SGD(model.parameters(), lr=cfg['maml']['inner_lr'])
            
            track_grads = not is_fomaml
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=track_grads, device=device) as (fmodel, diffopt):
                inner_loss, _ = vmc_loss(fmodel, task_H, cfg)
                diffopt.step(inner_loss)

                meta_loss, _ = vmc_loss(fmodel, task_H, cfg)
                
                if is_fomaml:
                    grads = torch.autograd.grad(meta_loss, fmodel.parameters())
                else:
                    grads = torch.autograd.grad(meta_loss, fmodel.parameters(time=0))

                for i, g in enumerate(grads):
                    if g is not None:
                        meta_grad_accumulator[i] += g
        
        for p, g in zip(model.parameters(), meta_grad_accumulator):
            p.grad = g / cfg['maml']['meta_batch_size']
        meta_optimizer.step()   
        
        print(f"Meta-Epoch [{meta_epoch+1}/{cfg['maml']['meta_epochs']}] done.")
            
    print("Meta-training finished.")
    return model.state_dict()


# --- FIX: Change the signature to accept a model instance ---
def evaluate(cfg, initial_params, model_instance, task_generator):
    num_test_tasks = cfg['evaluation']['num_test_tasks']
    finetune_steps = cfg['evaluation']['finetune_steps']
    
    energy_curves = np.zeros((finetune_steps, num_test_tasks))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model instance to the correct device once
    eval_model = model_instance.to(device)

    for i in tqdm(range(num_test_tasks), desc="Evaluating"):
        test_H = task_generator()
        
        # Load the initial state for the current evaluation run
        eval_model.load_state_dict(initial_params)
        eval_model.train()
        
        optimizer = SGD(eval_model.parameters(), lr=cfg['maml']['inner_lr'])

        for step in range(finetune_steps):
            optimizer.zero_grad()
            loss, energy = vmc_loss(eval_model, test_H, cfg)
            loss.backward()
            optimizer.step()
            energy_curves[step, i] = energy.item()
            
    return energy_curves