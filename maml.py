import torch
from torch.optim import SGD
import higher
from tqdm import tqdm
import numpy as np

from vmc import vmc_loss

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
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=track_grads) as (fmodel, diffopt):
                # Inner loop adaptation (t=1 step)
                inner_loss, _ = vmc_loss(fmodel, task_H, cfg['vmc']['n_samples'])
                diffopt.step(inner_loss)

                # Evaluate on adapted model to get meta-gradient
                meta_loss, _ = vmc_loss(fmodel, task_H, cfg['vmc']['n_samples'])
                
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
        
        if (meta_epoch + 1) % 10 == 0:
            print(f"Meta-Epoch [{meta_epoch+1}/{cfg['maml']['meta_epochs']}] done.")
            
    print("Meta-training finished.")
    return model.state_dict()

def run_sgd_baseline(cfg, model, task_generator):
    print("Starting baseline SGD training...")
    # For baseline, we train on a single, representative task
    task_H = task_generator() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = SGD(model.parameters(), lr=cfg['maml']['inner_lr'])
    
    # Train for a number of steps comparable to meta-training
    for _ in range(cfg['maml']['meta_epochs']):
        optimizer.zero_grad()
        loss, _ = vmc_loss(model, task_H, cfg['vmc']['n_samples'])
        loss.backward()
        optimizer.step()
        
    print("Baseline training finished.")
    return model.state_dict()

def evaluate(cfg, initial_params, model_class, task_generator):
    num_test_tasks = cfg['evaluation']['num_test_tasks']
    finetune_steps = cfg['evaluation']['finetune_steps']
    
    energy_curves = np.zeros((finetune_steps, num_test_tasks))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in tqdm(range(num_test_tasks), desc="Evaluating"):
        test_H = task_generator()
        eval_model = model_class.to(device)
        eval_model.load_state_dict(initial_params)
        eval_model.train()
        
        optimizer = SGD(eval_model.parameters(), lr=cfg['maml']['inner_lr'])

        for step in range(finetune_steps):
            optimizer.zero_grad()
            loss, energy = vmc_loss(eval_model, test_H, cfg['vmc']['n_samples'])
            loss.backward()
            optimizer.step()
            energy_curves[step, i] = energy.item()
            
    return energy_curves