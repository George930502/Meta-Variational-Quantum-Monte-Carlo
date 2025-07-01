import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
import numpy as np
import yaml
from hamiltonians import create_hilbert_space

class RBM(nn.Module):
    """Restricted Boltzmann Machine for variational wavefunction."""
    num_visible: int
    num_hidden: int
    param_dtype: jnp.dtype = jnp.float64
    
    def setup(self):
        # Initialize weights and biases similar to PyTorch version
        self.W = self.param(
            'W',
            lambda rng, shape, dtype: jax.random.normal(rng, shape, dtype) * 0.05,
            (self.num_visible, self.num_hidden),
            self.param_dtype
        )
        self.visible_bias = self.param(
            'visible_bias',
            lambda rng, shape, dtype: jnp.zeros(shape, dtype),
            (self.num_visible,),  # ← 注意要改成 vector 而不是 scalar
            self.param_dtype
        )
        self.hidden_bias = self.param(
            'hidden_bias',
            lambda rng, shape, dtype: jnp.zeros(shape, dtype),
            (self.num_hidden,),
            self.param_dtype
        )
    
    @nn.compact
    def __call__(self, x):
        # x shape: (..., num_visible)
        # Convert spins from {-1, +1} to {0, 1} for RBM
        v = (x + 1) / 2
        
        # Linear transformation: v @ W + h_bias
        linear = jnp.dot(v, self.W.reshape(-1, self.num_hidden)) + self.hidden_bias
        
        # Log-cosh activation for hidden units (equivalent to softplus for large values)
        hidden_contrib = jnp.sum(nn.softplus(linear), axis=-1)
        
        # Visible bias contribution
        visible_contrib = jnp.dot(v, self.visible_bias)
        
        return visible_contrib + hidden_contrib


class CNN(nn.Module):
    """Convolutional Neural Network for 2D lattice systems."""
    depth: int
    param_dtype: jnp.dtype = jnp.float64
    
    @nn.compact
    def __call__(self, x):
        # x shape: (..., num_spins)
        batch_shape = x.shape[:-1]
        num_spins = x.shape[-1]
        L = int(jnp.sqrt(num_spins))
        
        if L * L != num_spins:
            raise ValueError("CNN model requires a square number of spins.")
        
        # Reshape to 2D lattice: (..., 1, L, L)
        x_2d = x.reshape(*batch_shape, 1, L, L)
        
        # Convert from {-1, +1} to {0, 1}
        x_2d = (x_2d + 1) / 2
        
        # Convolutional layer
        x = nn.Conv(features=self.depth, 
                   kernel_size=(3, 3), 
                   padding='SAME',
                   param_dtype=self.param_dtype)(x_2d)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape(*batch_shape, -1)
        
        # Fully connected layer
        x = nn.Dense(features=num_spins, param_dtype=self.param_dtype)(x)
        
        # Sum over softplus activations
        return jnp.sum(nn.softplus(x), axis=-1)


def get_model(config):
    """Factory function to create the appropriate model based on configuration."""
    model_type = config['experiment']['model_type']
    num_spins = config['problem_params']['num_spins']
    
    if model_type == 'RBM':
        alpha = config['model_params']['rbm_alpha']
        num_hidden = int(alpha * num_spins)
        return RBM(num_visible=num_spins, num_hidden=num_hidden)
    elif model_type == 'CNN':
        depth = config['model_params']['cnn_depth']
        return CNN(depth=depth)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_netket_machine(config, hilbert_space):
    """Create a NetKet variational state with the specified model."""
    model = get_model(config)
    
    # Create the variational state
    sampler = nk.sampler.MetropolisLocal(hilbert_space, n_chains=config['vmc']['n_chains'], sweep_size=config['vmc']['mcmc']['decorrelation_sweeps'])

    key = jax.random.PRNGKey(config['seed'])
    sample = hilbert_space.random_state(key, 1)
    variables = model.init(key, sample)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=config['vmc']['n_samples'],
        n_discard_per_chain=config['vmc']['mcmc']['burn_in_sweeps'],
        variables=variables  
    )
    
    return vstate 

if __name__ == "__main__":
    config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)
    hilbert_space = create_hilbert_space(config['problem_params']['num_spins'])
    vstate = create_netket_machine(config, hilbert_space)
    print(vstate)