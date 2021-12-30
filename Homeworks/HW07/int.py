from scipy.integrate import quad
from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.Get_rank()

if rank == 0:
    start = MPI.Wtime()

f = lambda arg: np.sin(arg) ** 2

n_nodes = int(sys.argv[1])
a = 0.
b = np.pi / 2.

x = np.linspace(a, b, n_nodes) if rank == 0 else None
displ = None
count = np.empty(size, dtype=int)

if rank == 0:

    quot, rem = divmod(x.shape[0], size)

    for r in range(size):
        count[r] = quot + 1 if r < rem else quot

    displ = np.array([np.sum(count[:r]) for r in range(size)])

comm.Bcast(count, root=0)

x_part = np.zeros(count[rank], dtype=float)

comm.Scatterv([x, count, displ, MPI.DOUBLE], x_part, root=0)

part_sum = 0.

for i in range(1, x_part.shape[0]):
    dx = (x_part[i] - x_part[i - 1])
    part_sum += (f(x_part[i - 1]) + f(x_part[i])) * dx

res = comm.reduce(part_sum / 2, MPI.SUM, root=0)

if rank == 0:
    elapsed_time = MPI.Wtime() - start
    with open('times.csv', 'a+') as f:
        f.write(f'{size}, {elapsed_time}\n')

    # I, _ = quad(f, a, b)
    # error = np.abs(res - I)
    # with open(sys.argv[2], 'a+') as f:
    #     f.write(f'{n_nodes}, {error}\n')
    # print('pi =', round(4 * res, 2))
