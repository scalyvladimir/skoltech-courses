import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import tracemalloc
import imageio
import sys
import os

dir_name = 'pics/'

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.Get_rank()

# if rank == 0:
#     start = MPI.Wtime()

tracemalloc.start()

img = plt.imread(sys.argv[1]).transpose([1, 0, 2]) if rank == 0 else None

row_size = img.shape[1] if rank == 0 else None
row_size = comm.bcast(row_size, root=0)

col_size = img.shape[0] if rank == 0 else None
col_size = comm.bcast(col_size, root=0)

displ = None
count = np.empty(size, dtype=int)

if rank == 0:
    quot, rem = divmod(img.shape[0], size)

    for r in range(size):
        count[r] = quot + 1 if r < rem else quot
    displ = np.array([np.sum(count[:r]) for r in range(size)])

comm.Bcast(count, root=0)

img_part = np.zeros(count[rank] * row_size * 3, dtype=np.uint8)

if rank == 0:
    arg1, arg2 = count*row_size*3,  displ*row_size*3
    comm.Scatterv([np.ascontiguousarray(img), arg1, arg2, MPI.UINT8_T], img_part, root=0)
else:
    comm.Scatterv([np.ascontiguousarray(img), count, displ, MPI.UINT8_T], img_part, root=0)

img_part = img_part.reshape([count[rank], row_size, 3])

# if rank == 0:
#     images = []
#     if os.path.isdir(dir_name):
#         os.system(f'rm -rf {dir_name}')
#     os.mkdir(dir_name)

step_size = 1

for index, iter in enumerate(range(0, row_size, step_size)):
    comm.Barrier()
    # if rank == 0:
        # img_name = f'{str(index).zfill(4)}.png'
        # images.append(img_name)
        # plt.imsave(f'{dir_name}{img_name}', img.transpose([1, 0, 2]))

        # print(f'{iter / row_size * 100:.2f}% done')

    img_part = np.roll(img_part, axis=1, shift=step_size)

    img = np.ascontiguousarray(img)
    arg = displ * row_size * 3 if rank == 0 else row_size * 3
    comm.Gatherv(img_part, [img, count * row_size * 3, arg, MPI.UINT8_T], root=0)

    if rank == 0:
        img = img.reshape([col_size, row_size, 3])

if (rank == 0 and size == 1) or (rank == 1 and size != 1):
    with open(sys.argv[2], 'a+') as f:
        tmp = tracemalloc.get_traced_memory()[1]
        f.write(f'{size}, {tmp}\n')

# if rank == 0:
#
    # for i, im_name in enumerate(images):
    #     images[i] = imageio.imread(f'{dir_name}{im_name}')
    #
    # imageio.mimsave('shifted_image.gif', images)

    # elapsed_time = MPI.Wtime() - start
    # with open(sys.argv[2], 'a+') as f:
    #     f.write(f'{size}, {elapsed_time}\n')
