import warnings
from functools import wraps
from itertools import takewhile, count, islice

import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import diags, SparseEfficiencyWarning
from scipy.linalg import solve_banded

from polara.lib.sparse import arrange_indices
from polara.lib.tensor import ttm3d_seq, ttm3d_par
try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None


class SeqTFError(Exception):
    pass

def initialize_columnwise_orthonormal(dims, random_state=None):
    if random_state is None:
        random_state = np.random
    u = random_state.rand(*dims)
    u = np.linalg.qr(u, mode='reduced')[0]
    return u


def core_growth_callback(growth_tol):
    def check_core_growth(step, core_norm, factors):
        g_growth = (core_norm - check_core_growth.core_norm) / core_norm
        check_core_growth.core_norm = core_norm
        print(f'growth of the core: {g_growth}')
        if g_growth < growth_tol:
            print(f'Core is no longer growing. Norm of the core: {core_norm}.')
            raise StopIteration
    check_core_growth.core_norm = 0
    return check_core_growth


def sa_hooi(
        idx, val, shape, mlrank, attention_matrix, scaling_weights,
        max_iters = 10,
        parallel_ttm = False,
        growth_tol = 0.001,
        randomized=True,
        seed = None,
        iter_callback=None,
    ):

    assert valid_mlrank(mlrank)
    n_users, n_items, n_positions = shape
    r0, r1, r2 = mlrank
    
    tensor_data = idx, val, shape
    if not isinstance(parallel_ttm, (list, tuple)):
        parallel_ttm = [parallel_ttm] * len(shape)

    assert len(shape) == len(parallel_ttm)

    index_data = arrange_indices(idx, parallel_ttm)
    ttm = [ttm3d_par if par else ttm3d_seq for par in parallel_ttm]

    random_state = np.random if seed is None else np.random.RandomState(seed)
    u1 = initialize_columnwise_orthonormal((n_items, r1), random_state)
    uw = u1 * scaling_weights[:, np.newaxis]
    u2 = initialize_columnwise_orthonormal((n_positions, r2), random_state)
    ua = attention_matrix.dot(u2)

    if randomized:
        svd = randomized_svd
        svd_config = lambda rank: dict(n_components=rank, random_state=seed)
    else:
        svd = svds
        svd_config = lambda rank: dict(k=rank, return_singular_vectors='u')
    
    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol)
        
    
    for step in range(max_iters):
        ttm0 = ttm[0](*tensor_data, ua, uw, ((2, 0), (1, 0)), *index_data[0]).reshape(shape[0], r1*r2)
        u0, *_ = svd(ttm0, **svd_config(r0))

        ttm1 = ttm[1](*tensor_data, ua, u0, ((2, 0), (0, 0)), *index_data[1]).reshape(shape[1], r0*r2)
        u1, *_ = svd(ttm1, **svd_config(r1))
        uw = u1 * scaling_weights[:, np.newaxis]

        ttm2 = ttm[2](*tensor_data, uw, u0, ((1, 0), (0, 0)), *index_data[2]).reshape(shape[2], r0*r1)
        u2, ss, _ = svd(ttm2, **svd_config(r2))
        ua = attention_matrix.dot(u2)

        factors = (u0, u1, u2)
        try:
            iter_callback(step, np.linalg.norm(ss), factors)
        except StopIteration:
            break
    return factors


def exp_decay(decay_factor, n):
    return np.e**(-(n-1)*decay_factor)

def lin_decay(decay_factor, n):
    return n**(-decay_factor)



def attention_weights(decay_factor, cutoff, max_elements=None, exponential_decay=False, reverse=False):
    if (decay_factor == 0 or cutoff == 0) and (max_elements is None or max_elements <= 0):
        raise SeqTFError('Infinite sequence.')
    decay_function = exp_decay if exponential_decay else lin_decay
    weights = takewhile(lambda x: x>=cutoff, (decay_function(decay_factor, n) for n in count(1, 1)))
    if max_elements is not None:
        weights = islice(weights, max_elements)
    if reverse:
        return list(reversed(list(weights)))
    return list(weights)

def form_attention_matrix(size, decay_factor, cutoff=0, span=0, exponential_decay=False, reverse=False, format='csc', stochastic_axis=None, dtype=None):
    stochastic = stochastic_axis is not None
    span = min(span or np.iinfo('i8').max, size)
    weights = attention_weights(decay_factor, cutoff=cutoff, max_elements=span, exponential_decay=exponential_decay, reverse=reverse)
    diag_values = [np.broadcast_to(w, size) for w in weights]
    matrix = diags(diag_values, offsets=range(0, -len(diag_values), -1), format=format, dtype=dtype)
    if stochastic:
        scalings = matrix.sum(axis=stochastic_axis).A.squeeze()
        if stochastic_axis == 0:
            matrix = matrix.dot(diags(1./scalings))
        else:
            matrix = diags(1./scalings).dot(matrix)
    return matrix.asformat(format)


def generate_banded_form(matrix):
    matrix = matrix.todia()
    bands = matrix.data
    offsets = matrix.offsets
    num_l = (offsets < 0).sum()
    num_u = (offsets > 0).sum()
    return (num_l, num_u), bands[np.argsort(offsets)[::-1], :]


def generate_position_projector(attention_matrix, position_factors):
    shape, bands = generate_banded_form(attention_matrix.T)
    wl = solve_banded(shape, bands, position_factors)
    wr = attention_matrix.dot(position_factors)
    last_position_projector = wr @ wl[-1, :]
    return last_position_projector


def get_scaling_weights(frequencies, scaling=1.0):
    return np.power(frequencies, 0.5*(scaling-1.0))


def valid_mlrank(mlrank):
    prod = np.prod(mlrank)
    return all(prod//r > r for r in mlrank)