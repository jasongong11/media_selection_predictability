from scipy.sparse import csr_matrix, diags
import numpy as np
from itertools import product
from typing import Union, Tuple, Optional
import random

# code modified from https://github.com/maximtrp/mchmm
def transition_matrix(
        seq: Optional[Union[str, list, np.ndarray]] = None,
        ) -> csr_matrix:
    '''Calculate a transition frequency matrix.
    Parameters
    ----------
    seq : Optional[Union[str, list, numpy.ndarray]]
        Observations sequence. 
        It should be factorized into numerical representation of the sequence.
        such as [0, 1, 1, 2, 3,...]
    Returns
    -------
    matrix : numpy.ndarray
        Transition frequency matrix.
    '''

    seql = np.array(list(seq))
    n_states = seql.max()+1
    states = np.arange(n_states)
    # matrix = np.zeros((len(states), len(states)))

    matrix = csr_matrix((n_states, n_states), dtype=np.int16)
    
    for idx, track_a in enumerate(seql[:-1]):
        track_b = seql[idx+1]
        matrix[track_a, track_b] = matrix[track_a, track_b] + 1
    
#     for x, y in product(range(len(states)), repeat=2):
#         xid = np.argwhere(seql == states[x]).flatten()
#         yid = xid + 1
#         yid = yid[yid < len(seql)]
#         s = np.count_nonzero(seql[yid] == states[y])
#         matrix[x, y] = s

    return matrix

def transition_to_p_matrix(matrix):
    row_sums = matrix.sum(axis=1).A.ravel()
    
    if row_sums[-1] == 0 :
        row_sums[-1] = 1
        
    obs_row_sum_diag_matrix = diags(1/row_sums)
    observed_p_matrix = obs_row_sum_diag_matrix @ matrix
    
    return observed_p_matrix

def obs_p_matrix(seq):
    observed_matrix = transition_matrix(seq)
    # seql = np.array(list(seq))
    # n_states = seql.max()+1
    
#     # obs_row_totals = observed_matrix.sum(axis=1)
#     obs_row_sum_diag_matrix = diags(1/observed_matrix.sum(axis=1).A.ravel())

#     # observed transition probability matrix
#     # observed_p_matrix = observed_matrix / obs_row_totals
    
#     observed_p_matrix = obs_row_sum_diag_matrix @ observed_matrix
    
    observed_p_matrix = transition_to_p_matrix(observed_matrix)

    # filling in a row containing zeros with uniform p values
    # uniform_p = 1 / n_states
    # zero_row = np.argwhere(observed_p_matrix.sum(1) == 0).ravel()
    # observed_p_matrix[zero_row, :] = uniform_p
    
    return observed_p_matrix

def predict_markov(observed_p_matrix, old_state, ties="random"):
    
    max_index = np.argwhere(observed_p_matrix[old_state,:] == observed_p_matrix[old_state,:].max()).tolist()
    max_states = [state[1] for state in max_index]
    
    if len(max_states) == 1:
        return max_states[0]
    elif len(max_states) > 1:
        if ties == "random":
            return random.choice(max_states)
        elif ties == "first":
            return max_states[0]
        elif ties == "all":
            return np.array(max_states)

def total_evaluate_accuracy(sequence):
    correct_count = 0
    total_count = len(sequence)-1
    p_matrix = obs_p_matrix(sequence)
    
    for idx, old_state in enumerate(sequence[:-1]):
        pred_new_state = predict_markov(p_matrix, old_state)
        true_new_state = sequence[idx+1]
        if pred_new_state == true_new_state:
            correct_count += 1

    return correct_count/total_count


def loo_evaluate_accuracy(sequence, skip=1):
    correct_count = 0
    total_count = len(sequence)-1-skip
    
    n_states = sequence.max()+1
    observed_matrix = csr_matrix((n_states, n_states), dtype=np.int16)
    
    for idx, old_state in enumerate(sequence[:-2]):
        
        track_next = sequence[idx+1]
        observed_matrix[old_state, track_next] += 1
        
        if idx >= skip:
            n_current_states = sequence[:idx+2].max()+1
            observed_p_matrix = transition_to_p_matrix(observed_matrix[:n_current_states,:n_current_states])

            pred_new_state = predict_markov(observed_p_matrix, track_next)
            true_new_state = sequence[idx+2]
            if pred_new_state == true_new_state:
                correct_count += 1

    return correct_count/total_count