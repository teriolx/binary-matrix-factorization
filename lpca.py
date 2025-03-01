"""
https://github.com/schariya/exact-embeddings/blob/master/ExactEmbeddings.ipynb
"""
import numpy as np
import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize
from scipy.special import expit
from common import construct_adjacency_matrix, time_wrapper


def lpca_loss(factors, adj_s, rank): # adj_s = shifted adj with -1's and +1's
    n_row, n_col = adj_s.shape
    U = factors[:n_row*rank].reshape(n_row, rank)
    V = factors[n_row*rank:].reshape(rank, n_col)
    logits = U @ V
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -(prob_wrong * adj_s) @ V.T# / n_element
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element
    return loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))

def lpca_loss_fixed(factors, adj_s, rank, S): # adj_s = shifted adj with -1's and +1's
    # S is the fixed matrix of size rank/rank
    n_row, _ = adj_s.shape # adj_s is square
    U = factors[:n_row*rank].reshape(n_row, rank)
    V = S @ U.T
    logits = U @ V
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -(prob_wrong * adj_s) @ V.T# / n_element
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element
    print(loss)
    return loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))

clip_01 = lambda M : np.clip(M, a_min=0, a_max=1)

@time_wrapper
def decomposition_at_k(A, k, save_path=None, max_iter=2000):
    n, _ = A.shape

    factors = -1+2*np.random.random(size=2*n*k) # initalize uniformly on [-1,+1]
    res = scipy.optimize.minimize(lpca_loss, x0=factors, 
                            args=(-1 + 2*np.array(A.todense()), k), jac=True, method='L-BFGS-B',
                            options={'maxiter':max_iter})
    U = res.x[:n*k].reshape(n, k)
    V = res.x[n*k:].reshape(k, n)
    frob_error_norm = np.linalg.norm(clip_01(U@V) - A) / sp.sparse.linalg.norm(A)

    if save_path is not None:
        data = {'U':U, 'V':V, 'A': A}
        scipy.io.savemat(save_path, data)
    return frob_error_norm, res.nit


@time_wrapper
def decompose(A, max_k, max_iter=2000):
    n, _ = A.shape
    errors = []
    iters = []

    for k in range(1, max_k + 1):
        res = decomposition_at_k(A, k, max_iter)

        # if not res.success:
        #     # try with a lot higher num of iters if failed
        #     res = decomposition_at_k(A, k, 5000)
    
        U = res.x[:n*k].reshape(n, k)
        V = res.x[n*k:].reshape(k, n)
        frob_error_norm = np.linalg.norm(clip_01(U@V) - A) / sp.sparse.linalg.norm(A)

        errors.append(frob_error_norm)
        iters.append(res.nit)
    
    return errors, iters
