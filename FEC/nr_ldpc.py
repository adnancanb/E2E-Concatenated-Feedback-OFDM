import numpy as np
from scipy.io import loadmat
from scipy.linalg import circulant

def get_5g_ldpc_parity_matrix(info_len, coded_len):
    if coded_len <= 0:
        raise ValueError("coded_len must be positive.")
    coding_rate = info_len / coded_len
    bgs = loadmat('FEC/baseGraph.mat')
    if info_len <= 292:
        bgn = 2
    elif info_len <= 3824 and coding_rate <= 0.67:
        bgn = 2
    elif coding_rate <= 0.25:
        bgn = 2
    else:
        bgn = 1

    # add for consistency
    if bgn==1 and info_len>8448:
        raise ValueError("K is not supported by BG1 (too large).")

    if bgn==2 and info_len>3840:
        raise ValueError(
            f"K is not supported by BG2 (too large) k ={info_len}.")

    if bgn==1 and coding_rate<1/3:
        raise ValueError("Only coderate>1/3 supported for BG1. \
        Remark: Repetition coding is currently not supported.")

    if bgn==2 and coding_rate<1/5:
        raise ValueError("Only coderate>1/5 supported for BG2. \
        Remark: Repetition coding is currently not supported.")
    
    zc, i_ls, k_b = _sel_lifting(info_len, bgn)
    bg = bgs['BG{}S{}'.format(bgn, i_ls+1)]

    # total number of codeword bits
    n_ldpc = bg.shape[1] * zc
    # if K_real < K _target puncturing must be applied earlier
    k_ldpc = k_b * zc

    parity_matrix = prototype_to_parity(bg, zc)
    return parity_matrix, zc, k_ldpc, n_ldpc

def _sel_lifting(k, bgn):
    # lifting set according to 38.212 Tab 5.3.2-1
    s_val = [[2, 4, 8, 16, 32, 64, 128, 256],
            [3, 6, 12, 24, 48, 96, 192, 384],
            [5, 10, 20, 40, 80, 160, 320],
            [7, 14, 28, 56, 112, 224],
            [9, 18, 36, 72, 144, 288],
            [11, 22, 44, 88, 176, 352],
            [13, 26, 52, 104, 208],
            [15, 30, 60, 120, 240]]

    if bgn == 1:
        k_b = 22
    else:
        if k > 640:
            k_b = 10
        elif k > 560:
            k_b = 9
        elif k > 192:
            k_b = 8
        else:
            k_b = 6

    min_val = 100000
    z = 0
    i_ls = 0
    i = -1
    for s in s_val:
        i += 1
        for s1 in s:
            x = k_b *s1
            if  x >= k:
                if x < min_val:
                    min_val = x
                    z = s1
                    i_ls = i
    return z, i_ls, k_b

def prototype_to_parity(prototype, liftsize):
    shape = np.array(prototype.shape) * liftsize
    parity = np.zeros(shape)
    for i in range(prototype.shape[0]):
        for j in range(prototype.shape[1]):
            if prototype[i, j] >= 0:
                parity[i * liftsize : (i + 1) * liftsize, j * liftsize : (j + 1) * liftsize] = shift_mat(liftsize, prototype[i, j])
    return parity

def shift_mat(size, shift):
    row1 = np.zeros(size)
    shift = int(shift % size)
    row1[shift] = 1
    return circulant(row1).T
