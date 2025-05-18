import jax
import jax.numpy as jnp

def torus(x):
  return jnp.concatenate([jnp.sin(2*jnp.pi*x), jnp.cos(2*jnp.pi*x)], axis=-1)
