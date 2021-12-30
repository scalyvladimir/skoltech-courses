from mpi4py import MPI
# import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, exp

# def plot_specgram(specgram_data):
#     fig, ax = plt.subplots(figsize=(14, 11))
#
#
#     dims = [min(t) / (2 * pi), max(t) / (2 * pi), w[0], 2 * w[int(len(t) / 2) - 1]]
#
#     im = ax.imshow(specgram_data, aspect='auto', origin='lower', extent=dims)
#
#     fig.colorbar(im, orientation='vertical')
#
#     ax.set_ylim(0, 10)
#     ax.set_xlabel('window position', fontsize = 20)
#     ax.set_ylabel('frequency', fontsize = 20)
#     plt.savefig('spec.png')


def get_specgram(time, signal, rank, size, comm, nwindowsteps=1000):

    size_per_rank = int(nwindowsteps / size)

    if rank == size - 1:
        size_per_rank += nwindowsteps % size
        l, r = rank * size_per_rank, nwindowsteps - 1
    else:
        l, r = rank * size_per_rank, (rank + 1) * size_per_rank

    pos_list = np.linspace(-20, 20, nwindowsteps)

    data_part = np.empty([t.shape[0], size_per_rank])

    WIDTH = 1.5
    window_width = WIDTH * 2 * pi

    for pos_id, pos in enumerate(pos_list[l: r]):
        window_position = pos * 2 * pi
        window_function = exp(-(time - window_position) ** 2 / (2 * window_width ** 2))

        y_window = signal * window_function

        data_part[:, pos_id] = abs(np.fft.fft(y_window))

    data = comm.gather((data_part, rank) if size > 1 else data_part, root=0)

    if rank == 0:
        if rank == 0:
            if size > 1:
                data.sort(key=lambda tup: tup[1])
                data = [tup[0] for tup in data]
                data = np.hstack(data)
            else:
                np.array(data).squeeze()

        return data


NUM_CYCLES = 40

t = np.linspace(-NUM_CYCLES / 2 * 2 * pi, NUM_CYCLES / 2 * 2 * pi, 2*14)
y = np.sin(t) * exp(-t ** 2 / (NUM_CYCLES / 2) ** 2)
y += np.sin(3 * t) * exp(-(t - 5 * 2 * pi) ** 2 / (2 * 20 ** 2))
y += np.sin(5.5 * t) * exp(-(t + 10 * 2 * pi) ** 2 / (2 * 5 ** 2))
y += np.sin(4 * t) * exp(-(t - 7 * 2 * pi) ** 2 / (2 * 5 ** 2))
w = np.fft.fftfreq(len(y), d=(t[1] - t[0]) / (2 * pi))

N_REPS = 10

# start = float(0)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = MPI.Wtime()

for _ in range(N_REPS):

    get_specgram(t, y, rank, size, comm, nwindowsteps=1000000)

    # if rank == 0:
    # accumulator += elapsed_time

comm.barrier()
elapsed_time = MPI.Wtime() - start

MPI.Finalize()

if rank == 0:
    with open('results.csv', 'a+') as f:
        f.write(f'{size}, {elapsed_time / N_REPS:.3f}\n')
