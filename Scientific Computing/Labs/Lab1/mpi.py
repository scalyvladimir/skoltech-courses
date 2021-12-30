from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD 
rank = comm.Get_rank() 

M = 100
def getPartialSum(start, end):
    a = np.arange(start, end)
    return np.sum(1./(a*a))

s = getPartialSum(1+rank*M, 1+(rank+1)*M)
print ('Process', rank, 'found partial sum from term', 1+rank*M, 'to term', 1+(rank+1)*M-1, ': ', s )

# process 1 sends its partial sum to process 0
if rank == 1:
    comm.send(s, dest=0) 
    
# process 0 receives the partial sum from process 1, adds to its own partial sum
# and outputs the result    
elif rank == 0: 
    s_other = comm.recv(source=1)
    s_total = s+s_other
    print ('total partial sum =', s_total)
    print ('pi_approx =', np.sqrt(6*s_total))
    
print ('Process', rank, 'finished')
