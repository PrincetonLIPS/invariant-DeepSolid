"""
    Additional evaluations for samples from trained DeepSolid-based wavefunctions.

    Isolated ground state energy taken from https://arxiv.org/pdf/1909.02487, similar to DeepSolid
"""
import jax, time
import jax.numpy as jnp
import numpy as np 
from utils.addkeys import split_x_key
from DeepSolid import constants

Ha_to_eV = 27.2114079527

# OBSOLETE: This number can be replaced by whichever reference energy one wants to use, although for a small supercell size, the comparison may not be valid without appropriate corrections. We chose not to report cohesive energy in the final paper.
atom_Ha_dict = {
    'C': -37.84471,
    'Li': -7.47798,
    'H': -13.59844 / Ha_to_eV,
}

base_key = jax.random.PRNGKey(int(1e6 * time.time()))

def compute_cohesive_energy(supercell_e_Ha, atoms: list):
    '''
        Args:
            supercell_e_Ha: energy of the supercell in Hartree
            atoms: a list of atoms of the supercell, obtained from cfg.system.pyscf_cell.atom
    '''
    isolated_e_Ha = np.sum([atom_Ha_dict[atom[0]] for atom in atoms])
    cohesive_eV = (supercell_e_Ha - isolated_e_Ha) * Ha_to_eV
    return cohesive_eV

def make_symm_measure_computation(measure_f, f, need_f_key, same_key: bool = False):
    '''
        Returns function for computing Var([reference net](x))/([current net](x))
    '''
    def compute_ratio(params, x_with_key):
        '''
            compute [reference net](x) / [current net](x)
            x_with_key: jnp.ndarray of shape (3N+1,)
        '''
        assert len(x_with_key.shape) == 1
        x, key_int = split_x_key(x_with_key)
        key = jax.random.fold_in(base_key, key_int)
        ref_key, sub_key = jax.random.split(key)
        ref_key = ref_key[0]
        if same_key is False:
            sub_key = sub_key[0]
        else:
            sub_key = ref_key
            
        ref_output = measure_f(params, x, ref_key)
        if need_f_key:
            output = f(params, x, sub_key)
        else:
            output = f(params, x)
        return jnp.exp(ref_output - output)
    
    def compute_symmetry_measure(params, data_and_keys):
        ratios = jax.vmap(
                    compute_ratio, in_axes=(None,0,), out_axes=0
                 )(params, data_and_keys)
        mean = constants.pmean_if_pmap(jnp.mean(ratios), axis_name=constants.PMAP_AXIS_NAME)
        var = constants.pmean_if_pmap(jnp.mean(jnp.abs(ratios - mean)**2), axis_name=constants.PMAP_AXIS_NAME)
        return mean, var

    return compute_symmetry_measure