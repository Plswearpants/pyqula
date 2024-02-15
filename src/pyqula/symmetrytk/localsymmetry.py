import numpy as np


import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit
from jax import grad


def permutations(H,nk=20):
    """Return all the symmetries of a Hamiltonian that arise from permuting
    sites"""
    ks = np.random.random((nk,3)) # generate random kpoints
    hk = H.get_hk_gen() # get the function generator
    ms = [hk(k) for k in ks] # generate all the matrices
    n = ms[0].shape[0] # size of the matrix
    U = np.random.random((n,n)) - 0.5 # random matrix
    Ua = U.reshape(n*n) # make square
    # now define function and gradient
    error_jax = jit(error_master) # jit jax function for error
    jac_error_jax = jit(grad(error_master,argnums=0)) # jit jax gradient
    #####
    def fun(Ua):
        return error_jax(Ua,ms) # return function
    def jac_fun(Ua):
        return jac_error_jax(Ua,ms) # return jacobian
    from scipy.optimize import minimize
    tol = 1e-12
    result = minimize(fun,Ua,tol=tol,jac=jac_fun)
    Ua = result.x
    U = Ua.reshape((n,n))
    print(np.round(U,2),fun(Ua))
    return U # return the matrix



def error_master(Ua,ms):
    """Compute the error of how much a potential unitary U
    commutes with the matrices ms, and how unitary is"""
    n = ms[0].shape[0] # dimension
    U = Ua.reshape((n,n)) # make square
    iden = jnp.identity(n) # identity
    err = 0. # initialize
    for m in ms: # loop over matrices
        dm = m@U - U@m # difference in the commutation
        dmd = jnp.conjugate(dm.T) # dagger
        err = err + jnp.trace(dm@dmd) # trace of the matrix
    # now error in the unitarity
    dU = iden - U@jnp.conjugate(U.T) 
    err = err + jnp.trace(dU@dU) # trace of the matrix
    # now error from being a permutation
    for iu in U:
        iu2 = iu*iu
        erri = jnp.sum(iu2*jnp.log(iu2))
        err = err + erri*erri
    return err.real





def retain_independent(Ms,tol=1e-4):
    """Return linearly independent matrices. The parameter tol
    gives the threshold for linear independence"""
    from numpy.linalg import matrix_rank
    n = Ms[0].shape[0] # input are matrices, get their dimension
    M = [m.reshape((n*n)] for m in Ms] # reshape as vectors
    dim = len(M) # number of matrices
    LI=[] # output matrices
    for i in range(dim):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])
        if matrix_rank(tmp,tol=tol)>len(LI):
            LI.append(M[i])
    LIo = [m.reshape((n,n)) for m in LI] # reshape as matrices
    return LI



