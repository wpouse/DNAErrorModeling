import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import itertools
from measurementModel import genDNA

class DNAExtract:
    def __init__(self, measured_sequences, incorp_err, cleave_err, dark_err):
        self.measured_sequences = measured_sequences
        self.incorp_err = incorp_err
        self.cleave_err = cleave_err
        self.dark_err = dark_err
        self.init_DNA = self._extract_initDNA()
        return


    def DNA_to_seq(self, DNA_string):
        seq_lookup = {'A': 0, 'T': 1, 'G':2, 'C': 3}        
        return np.array([seq_lookup[c] for c in DNA_string])

    def seq_to_DNA(self, sequence):
        DNAstring_lookup = {0: 'A', 1:'T', 2:'G', 3:'C'}
        return ''.join([DNAstring_lookup[i] for i in sequence])
    
    def _extract_initDNA(self):
        hist = self.get_hist(self.measured_sequences, 1)
        return np.array(pd.DataFrame(data = hist).idxmax())
    
    def gen_windowed_seq(self, seq, L, S=1):
        # Window len = L, Stride len/stepsize = S
        nrows = ((seq.shape[1]-L)//S)+1
        n = seq.strides[1]
        return np.lib.stride_tricks.as_strided(seq, shape=(seq.shape[0],nrows,L), strides=(seq.strides[0],S*n,n))

    def get_hist(self, measured_seq, window_length = 1):
        reference_seq = np.array(list(itertools.product([0, 1, 2, 3], repeat=window_length)))
        windowed_seq = self.gen_windowed_seq(measured_seq, window_length, 1)
        return (reference_seq[None, :,:] == windowed_seq[:, :, None, :]).all(axis=3).sum(axis=0).T

    def histogram_err(self, hist1, hist2):
        return np.sum((hist1-hist2)**2)

    def get_avg_err(self, sequence, comparison_hist, num_avg, window_length):
        err = []
        for _ in range(num_avg):
            new_hist = self.get_hist(genDNA(np.append(sequence, np.random.randint(0,4, len(sequence))), self.incorp_err, self.cleave_err, self.dark_err, self.measured_sequences.shape[0]), window_length = window_length)
            new_err = self.histogram_err(new_hist, comparison_hist)
            err.append(new_err)
        return np.median(err)

    def base_seq_extract(self, window, num_err_avg, num_steps = 1000, fixed_base_num = 0):

        errors = np.zeros(num_steps+1)

        curr_DNA = self.init_DNA
        meas_hist = self.get_hist(self.measured_sequences, window_length = window)

        num_sequences = self.measured_sequences.shape[0]

        errors[0] = self.get_avg_err(curr_DNA, meas_hist, num_avg = num_err_avg, window_length = window)
        curr_err = errors[0]


        if fixed_base_num < 0:
            fixed_base_num = 0
        for i in range(num_steps):
            base_loc = np.random.randint(fixed_base_num, self.measured_sequences.shape[1])

            next_DNA = np.copy(curr_DNA)

            #Choose whether to try a deletion or insertion
            if np.random.randint(0,2) == 0:
                next_DNA = np.append(np.delete(next_DNA, base_loc), np.random.choice([j for j in range(4)]))
            else:
                next_DNA = np.insert(next_DNA, base_loc, np.random.choice([j for j in range(4)]))[:len(curr_DNA)]
            

            next_err = self.get_avg_err(next_DNA, meas_hist, num_avg = num_err_avg, window_length = window)
            if (next_err < curr_err) or (np.random.rand(1)[0] < np.exp(-(next_err-curr_err)/10)):
                curr_err = next_err
                curr_DNA = next_DNA

            errors[i+1] = curr_err
        self.pred_DNA = curr_DNA
        self.errors = errors
        return self.pred_DNA
    
    def base_seq_extract_iterative(self, window, num_err_avg, num_optimize, num_steps = 1000):
        """Optimizes num_optimize elements at a time since errors are more likely at end of sequences"""
        
        #Split up steps with log spacing so less steps are used in first part of sequence
        step_nums = np.diff((np.logspace(0, np.log10(num_steps), self.measured_sequences.shape[1]//num_optimize +1)).astype(int))
        errors = []
        self.pred_DNA = self.init_DNA
        for i, step_num in enumerate(step_nums):
            if i == len(step_nums)-1:
                _DNAex = DNAExtract(self.measured_sequences[:, :], self.incorp_err, self.cleave_err, self.dark_err)
            else:
                _DNAex = DNAExtract(self.measured_sequences[:, :(i+1)*num_optimize], self.incorp_err, self.cleave_err, self.dark_err)
            _DNAex.init_DNA[:(i)*num_optimize] = self.pred_DNA[:(i)*num_optimize]
            _DNAex.base_seq_extract(window, num_err_avg, step_num, fixed_base_num = i*num_optimize-2) #Use -2 for fixed_base_num to ensure some overlap
            errors.append(_DNAex.errors)
            self.pred_DNA[:(i+1)*num_optimize] = _DNAex.pred_DNA
            
        
        return self.pred_DNA
    
    def DL_dist_nosub(self,s1, s2):
        d = {}
        lenstr1 = len(s1)
        lenstr2 = len(s2)
        for i in range(-1,lenstr1+1):
            d[(i,-1)] = i+1
        for j in range(-1,lenstr2+1):
            d[(-1,j)] = j+1

        for i in range(lenstr1):
            for j in range(lenstr2):
                if s1[i] == s2[j]:
                    #cost = 0
                    d[(i,j)] = min(
                               d[(i-1,j)] + 1, # deletion
                               d[(i,j-1)] + 1, # insertion
                               d[(i-1,j-1)], # two characters are the same
                              )
                else:
                    #cost = 10
                    d[(i,j)] = min(
                                   d[(i-1,j)] + 1, # deletion
                                   d[(i,j-1)] + 1, # insertion
                                  )
                #Not considering transpositions
    #            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
    #                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

        return d[lenstr1-1,lenstr2-1]
