import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start = MPI.Wtime()

NUM_OF_STEPS = 10000
INITIAL_STATE = 0.1
M = 1000
N = 500

r_list = np.linspace(1, 10, M)


def get_next_step(r, x):
    return r * x * (1. - x)


size_per_rank = int(M / size)

if rank == size - 1:
    size_per_rank += M % size
    l, r = rank * size_per_rank, M - 1
else:
    l, r = rank * size_per_rank, (rank + 1) * size_per_rank

data_part = np.empty([size_per_rank, M])

for index, r in enumerate(r_list[l: r]):
    x = [INITIAL_STATE]
    for _ in range(NUM_OF_STEPS):
        x.append(get_next_step(r, x[-1]))

    data_part[index] = x[N: N + M]

data = comm.gather((data_part, rank) if size > 1 else data_part, root=0)

if rank == 0:
    if size > 1:
        data.sort(key=lambda tup: tup[1])
        data = np.array(data)[:, :-1].squeeze()
        data = np.concatenate(data, axis=0)
    else:
        data = np.array(data).squeeze()

if rank == 0:
    elapsed_time = MPI.Wtime() - start
    with open('results.csv', 'a+') as f:
        f.write(f'{size}, {elapsed_time}\n')

MPI.Finalize()

if rank == 0:
    print(f'{size} processes done')

#   fig, ax = plt.subplots(figsize=(15, 15))

#   fig.set_facecolor('#846075')

#   ax.axis([min(r_list), 5, 0, 1 + 0.1])
#   ax.set_facecolor('#4A6C6F')

#   ax.set_xlabel('r')
#   ax.set_ylabel('population equlibrium')
#   l, = ax.plot([], [], '.', color='#D7DEDC')


#   print(data.shape, data)
#   t = [r * np.ones(data.shape[0]) for r in r_list]

# dir_name = 'pics'

# os.system(f'rm -rf {dir_name}')
# os.mkdir(dir_name)

# images = []

# for i, _ in enumerate(t):
#     index = str(i).zfill(4)
#     imgname = f'{dir_name}/bif_pic_{index}'
#     images.append(imgname)

#     ax.scatter(t[:i], data[:i], c='white')
#     plt.savefig(images[-1])
