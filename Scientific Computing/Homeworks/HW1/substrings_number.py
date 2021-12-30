import numpy as np


def invPerm(p):
    """Invert the permutation p"""
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    print('permutation: ', s)
    return s


def getSuffixArray(A):
    A = np.array(list(A))

    N = len(A)
    # num of times word could be divided by 2
    M = int(np.ceil(np.log2(N))) + 1

    # positions of sorted length-(2**m) sequences in A
    P = np.zeros((M, N), dtype=int)

    # rank (0, 1, etc.) of sorted length-(2**m) sequences after sorting
    Q = np.zeros((M, N), dtype=int)

    # rank of sorted length-(2**m) sequences at its starting position in A;
    # padded by 0 on the right
    R = np.zeros((M, 3 * N), dtype=int)

    P[0] = np.argsort(A)
    # A[P[0]] <=> sorted A
    Q[0][1:] = np.cumsum(A[P[0]][1:] != A[P[0]][:-1])
    R[0][:N] = Q[0][invPerm(P[0])]

    print('P:', P[0], 'Q:', Q[0], 'R:', R[0])
    m = 0
    for m in range(1, M):
        offset = 2 ** (m - 1)
        r = np.lexsort((R[m - 1, P[m - 1] + offset], R[m - 1, P[m - 1]]))
        P[m] = P[m - 1][r]
        # m'th rank increases iff (m-1)'th rank increases at least for one element of the pair
        Q[m][1:] = np.cumsum(np.logical_or(R[m - 1][P[m]][1:] != R[m - 1][P[m]][:-1],
                                           R[m - 1][P[m] + offset][1:] != R[m - 1][P[m] + offset][:-1]))
        R[m][:N] = Q[m][invPerm(P[m])]

        # early stopping if suffixes already fully sorted (max rank is N-1)
        print('P:', P[m], 'Q:', Q[m], 'R:', R[m])
        if Q[m][-1] == N - 1:
            break

    SA = P[m]
    return SA, P[:m + 1], Q[:m + 1], R[:m + 1]


def getLCP(SA, R):
    (M, N) = R.shape
    LCP = np.zeros((len(SA) - 1,), dtype=int)
    for m in range(M - 1)[::-1]:
        t = (R[m][SA[1:] + LCP] == R[m][SA[:-1] + LCP]).astype(int)
        LCP += (2 ** m) * t
    return LCP


size = int(input())
for _ in range(size):
    A = input() + '$'
    SA, _, _, R = getSuffixArray(A)
    LCP = getLCP(SA, R)

    arythm_sum = (1 + len(A) - 1) / 2 * (len(A) - 1)

    print(int(arythm_sum - sum(LCP)))
