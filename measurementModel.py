import numpy as np
import itertools

def genDNA(base_strand, incorp_err, cleave_err, dark_err, num_copies = 100):
    num_sequencing = int(len(base_strand)//2)
    measured_DNA = np.zeros((num_copies, num_sequencing), dtype = int) - 1 #set to -1 to not confuse with 0 (A)
    counter = np.zeros(num_copies, dtype=int) #Keep track of the current position of each strand
    DNA_array = np.tile(base_strand, (num_copies,1))
    for t in range(num_sequencing): 
        rand_numbers = np.random.rand(num_copies)
        counter -= (rand_numbers < incorp_err).astype(int)

        rand_numbers = np.random.rand(num_copies)
        counter += (rand_numbers < dark_err).astype(int)
        counter[counter < 0] = 0 #Can't have n-1 error on first step?

        measured_DNA[:, t] = DNA_array[range(num_copies), counter]
        counter += 1
    return measured_DNA

def gen_windowed_seq(seq, L, S=1):
    # Window len = L, Stride len/stepsize = S
    nrows = ((seq.shape[1]-L)//S)+1
    n = seq.strides[1]
    return np.lib.stride_tricks.as_strided(seq, shape=(seq.shape[0],nrows,L), strides=(seq.strides[0],S*n,n))

def get_hist(measured_seq, window_length = 1):
    reference_seq = np.array(list(itertools.product([0, 1, 2, 3], repeat=window_length)))
    windowed_seq = gen_windowed_seq(measured_seq, window_length, 1)
    return (reference_seq[None, :,:] == windowed_seq[:, :, None, :]).all(axis=3).sum(axis=0).T