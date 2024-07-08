import pandas as pd
import matplotlib.pyplot as plt

import logomaker

def count_sixmers(seq, six_mers_counts):
        if type(seq) == float or len(seq) < 6:
            return
        for i in range(len(seq)-5):
            sixmer = seq[i:i+6]
            if sixmer not in six_mers_counts:
                six_mers_counts[sixmer] = 1
            else:
                six_mers_counts[sixmer] += 1
        return six_mers_counts

def count_kmers(seqs, k):
    k_mers_counts = {}
    for seq in seqs:
        if type(seq) == float:
            continue
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            if kmer not in k_mers_counts:
                k_mers_counts[kmer] = 1
            else:
                k_mers_counts[kmer] += 1
    return k_mers_counts

def update_kmers_counts(k_mers_counts, seqs, k):
    for seq in seqs:
        if type(seq) == float:
            continue
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            if kmer not in k_mers_counts:
                k_mers_counts[kmer] = 1
            else:
                k_mers_counts[kmer] += 1
    return k_mers_counts

def update_kmer_counts_per_tpm(k_mer_tpms, row, k):
    
    seq = row['upstream200']
    if type(seq) == float:
        return k_mer_tpms
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        for condition in [cond for cond in row.keys() if 'tpm' in cond]:
            stress = condition.split('_')[0]
            if kmer not in k_mer_tpms[stress]:
                k_mer_tpms[stress][kmer] = 0
            if row[condition] == 0:
                continue
            k_mer_tpms[stress][kmer] += 1/row[condition]  # to get sequences per TPM unit: 1/tpm
    return k_mer_tpms


def plot_logos(species_full_count_data):
    for spec_id in species_full_count_data:
        logo = logomaker.Logo(species_full_count_data[spec_id][1], color_scheme='classic')
        logo.ax.set_title(spec_id)
        plt.show()

def plot_sixmer_counts(species_full_count_data):
    for spec_id in species_full_count_data:
        species_full_count_data[spec_id][0].head(30).plot(kind='bar')
        plt.title(spec_id)
        plt.show()
            
        
            