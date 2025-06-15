"""
https://github.com/schariya/exact-embeddings/blob/master/ExactEmbeddings.ipynb
"""
import numpy as np
import scipy as sp
import scipy.io, scipy.optimize
from scipy.special import expit
from common import time_wrapper, measure_encoding_similarity


def similarity_loss(A, encodings):
    similarities = measure_encoding_similarity(A, encodings)

    loss = 0
    for d, distances in similarities.items():
        for s in distances:
            loss += s / (d + 1) ** 2

    return loss


def lpca_loss(factors, adj_s, rank, config):
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
        config:    dictionary with additional configurable parameters
                   - fixed_V: in the case of fixed optimization, 
                    the matrix that is fixed and from which the product 
                    with factors will be calculated to obtain the loss
                   - is_single: indicator whether the factors only represent one matrix, 
                    which is then transposed and multiplied with itself to 
                    calculate the loss from
                   - p_lambda: parameter for combined loss weighting 

      Returns:
        int: calculated loss between the decomposed matrix and the adjacency matrix
    """
    # adj_s = shifted adj with -1's and +1's
    n, _ = adj_s.shape

    U = factors[:n*rank].reshape(n, rank)
    
    if config["fixed_V"] is not None:
        V = config["fixed_V"]
    elif config["is_single"]:
        V = U.T
    else:
        V = factors[n*rank:].reshape(rank, n) 
    
    if config["is_single"]:
        print("here")
        A = clip_01(adj_s)
        T_0 = np.exp(-(A * (U).dot(U.T)))
        T_1 = (np.ones((n, n)) + T_0)
        T_2 = (T_1 ** -1)
        loss = -np.sum((np.log(T_2)).dot(np.ones(n)))
        U_grad = -(np.ones(n) * 2)[:, np.newaxis] * (((((T_1 ** -2) * T_0) * A) / T_2).dot(U))
        return loss, U_grad.flatten()
    
    # alternative gradient definitions
    # A = adj_s
    # T_0 = np.exp(-(A * (U).dot(V)))
    # T_1 = (np.ones((n, n)) + T_0)
    # T_2 = (T_1 ** -1)
    # loss = -np.sum((np.log(T_2)).dot(np.ones(n)))
    # U_grad = -(((((T_1 ** -2) * T_0) * A) / T_2)).dot(V.T)
    # V_grad = -(U.T).dot(((((T_1 ** -2) * T_0) * A) / T_2))
    
    logits = U @ V
    prob_wrong = expit(-logits * adj_s)
    loss = (np.logaddexp(0,-logits*adj_s)).sum()# / n_element    
    U_grad = -((prob_wrong) * adj_s) @ V[:, :n].T# / n_element
    V_grad = -U.T @ (prob_wrong * adj_s)# / n_element

    sim_loss = 0
    if config["p_lambda"] != 0:
        sim_loss = similarity_loss(clip_01(adj_s), np.hstack((U, V.T)))

    if config["fixed_V"] is not None or config["is_single"]:
        assert loss + config["p_lambda"] * sim_loss == loss
        return loss + config["p_lambda"] * sim_loss, U_grad.flatten()
    
    return loss + config["p_lambda"] * sim_loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))  


clip_01 = lambda M : np.clip(M, a_min=0, a_max=1)

@time_wrapper
def decomposition_at_k(A, 
                       k, 
                       config,
                       save_path=None):
    n, _ = A.shape


    size = 2*n*k
    if config["fixed_V"] is not None or config["is_single"]:
        # only one matrix for optimization
        size = n*k
    
    # initalize uniformly on [-1,+1]
    factors = -1+2*np.random.random(size=size)

    bounds_list = None
    if config['bounds'] is not None:
        bounds_list = [config['bounds'] for _ in range(len(factors))]
    
    res = scipy.optimize.minimize(lambda x, adj_s, rank: lpca_loss(x, adj_s, rank, config), 
                                  x0=factors, 
                                  args=(-1 + 2*np.array(A.todense()), k),
                                  jac=True,
                                  method='L-BFGS-B',
                                  options={'maxiter': config["max_iter"]},     
                                  bounds=bounds_list)
    
    U = res.x[:n*k].reshape(n, k) 

    if config["fixed_V"] is not None:
        V = config["fixed_V"]
    elif config["is_single"]:
        V = U.T
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
