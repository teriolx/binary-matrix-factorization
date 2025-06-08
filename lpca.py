"""
https://github.com/schariya/exact-embeddings/blob/master/ExactEmbeddings.ipynb
"""
import numpy as np
import scipy as sp
import scipy.io, scipy.optimize
from scipy.special import expit
from common import time_wrapper, measure_encoding_similarity


def normalize_loss(loss1, loss2):
    total = abs(loss1) + abs(loss2)
    loss1_norm = loss1 / total
    loss2_norm = loss2 / total

    return 0.8 * loss1_norm + 0.2 * loss2_norm


def similarity_loss(A, encodings):
    similarities = measure_encoding_similarity(A, encodings)

    loss = 0
    for d, distances in similarities.items():
        for s in distances:
            loss += s / (d + 1) ** 2

    return loss


def lpca_loss(factors, adj_s, rank, V_fixed=None, is_single=False, fixed_k=None):
    """
      Calculates the loss used for optimizing the U, V decomposition.
      In the classic case, both U and V are optimized, where the 
      factors corresponds to the nummber of elements in each matrix. 
      In the case of fixing one matrix, the factors are only optimized
      as one matrix, and the other matrix is given to calculate the loss.
      Lastly, in the case of opitmizing a single matrix, the loss is
      calculated by transposing the matrix to obtain the product.

      Args:
        factors:   flattened values of the matrices being optimized, the
                   loss is calcualted from these values once they are arranged
                   into the desired shape based on the case (see above)
        adj_s:     graph adjacency matrix shifted -1's and +1's
        rank:      desired size of the matrices
        V_fixed:   in the case of fixed optimization, the matrix that is fixed
                   and from which the product with factors will be calculated
                   to obtain the loss
        is_single: indicator whether the factors only represent one matrix, 
                   which is then transposed and multiplied with itself to 
                   calculate the loss from

      Returns:
        int: calculated loss between the decomposed matrix and the adjacency matrix
    """
    # adj_s = shifted adj with -1's and +1's
    n, _ = adj_s.shape

    U = factors[:n*rank].reshape(n, rank)
    
    if V_fixed is not None:
        V = V_fixed
    elif is_single:
        V = U.T
    elif fixed_k is not None:
        V = fixed_k @ U.T
    else:
        V = factors[n*rank:].reshape(rank, n) 

    logits = U @ V[:, :n]
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -(prob_wrong * adj_s) @ V[:, :n].T# / n_element
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element

    sim_loss = 0
    if fixed_k is not None:
        sim_loss = similarity_loss(clip_01(adj_s), U)

    if V_fixed is not None or is_single or fixed_k is not None:
        return loss + sim_loss, U_grad.flatten(), V_grad.flatten()

    return loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))  

clip_01 = lambda M : np.clip(M, a_min=0, a_max=1)

@time_wrapper
def decomposition_at_k(A, 
                       k, 
                       save_path=None, 
                       fixed_V=None, 
                       max_iter=2000, 
                       bounds=None, 
                       is_single=False, 
                       fixed_k=None):
    n, _ = A.shape

    # initalize uniformly on [-1,+1]
    size = 2*n*k
    if fixed_V is not None or is_single or fixed_k is not None:
        # only one matrix for optimization
        size = n*k
    factors = -1+2*np.random.random(size=size)
    
    bounds_list = [bounds for _ in range(len(factors))] if bounds is not None else None
    res = scipy.optimize.minimize(lambda x, adj_s, rank: lpca_loss(x, adj_s, rank, fixed_V, is_single, fixed_k), 
                                  x0=factors, 
                                  args=(-1 + 2*np.array(A.todense()), k), 
                                  jac=True,
                                  method='L-BFGS-B',
                                  options={'maxiter':max_iter},
                                  bounds=bounds_list)
    
    U = res.x[:n*k].reshape(n, k) 

    if fixed_V is not None:
        V = fixed_V
    elif is_single:
        V = U.T
    elif fixed_k is not None:
        V = fixed_k @ U.T
    else:
        V = res.x[n*k:].reshape(k, n)
    
    A_reconstructed = clip_01(U@V)

    frob_error_norm = np.linalg.norm(A_reconstructed[:,:n] - A) / sp.sparse.linalg.norm(A)

    data = {'U':U, 'V':V, 'A': A}

    if save_path is not None:
        scipy.io.savemat(save_path, data)
    return frob_error_norm, res.nit, data


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
