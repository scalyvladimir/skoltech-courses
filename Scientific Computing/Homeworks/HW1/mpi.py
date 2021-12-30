from mpi4py import MPI
from os import listdir
from os.path import isfile, join

my_path = 'C:\\Users\\ghost\\Desktop\\test'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

files_list = [f for f in listdir(my_path) \
              if isfile(join(my_path, f)) \
              and f.endswith('.txt')]

id = rank - 1 if size > 1 else 0
local_max = 0

while 0 <= id < len(files_list):
    file_name = f'{my_path}\\{files_list[id]}'

    num_lines = sum(1 for line in open(file_name))

    local_max = max(num_lines, local_max)
    id += size

global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

if rank == 0:
    print(f'The biggest text file has {global_max} lines')