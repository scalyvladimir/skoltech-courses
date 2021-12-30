from multiprocess import Pool, cpu_count
from tqdm.auto import tqdm


def run_parallel(f, N, n_cores=None):
    if n_cores is None:
        n_cores = max(cpu_count() - 2, 1)
	
    res = []
    with Pool(processes=n_cores) as p:
        for ans in tqdm(p.imap_unordered(f, N), 
                        total=len(N)):
            res.append(ans)
    return res
