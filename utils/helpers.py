import torch
from torch.utils.data import Dataset

# dictionary for eqach base and its complement
dna_complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

# dictionary for each species and its file name in the data sequences folder
species_to_data_sequences = {
    "Vibrio cholerae O1 biovar El Tor str. N16961": "GCF_000006745",
    "Pseudomonas aeruginosa PAO1": "GCF_000006765",
    "Neisseria gonorrhoeae FA 1090": "GCF_000009465",
    "Legionella pneumophila subsp. pneumophila Philadelphia 1": "GCF_000008485",
    "Borreliella burgdorferi B31": "GCF_000008685",
    "Helicobacter pylori G27": "GCF_000021165",
    "Staphylococcus aureus MRSA252": "GCF_002085695",
    "Burkholderia pseudomallei K96243": "GCF_000011545",
    "Haemophilus influenzae 86-028NP": "GCF_000012185",
    "Escherichia coli EPEC 0127:H6 E2348/69": "GCF_000210475",
    "Streptococcus agalactiae NEM316": "GCF_001021955",
    "Campylobacter jejuni subsp. jejuni 81-176": "GCF_000015525",
    "Mycobacterium tuberculosis H37Ra": "GCF_000016145",
    "Klebsiella pneumoniae subsp. pneumoniae MGH 78578": "GCF_000016305",
    "Aggregatibacter actinomycetemcomitans D7S-1": "GCF_000163615",
    "Francisella tularensis subsp. holarctica FSC200": "GCF_000168775",
    "Enterococcus faecalis OG1RF": "GCF_000172575",
    "Listeria monocytogenes EGD-e": "GCF_000196035",
    "Salmonella enterica subsp. enterica serovar Typhimurium SL1344": "GCF_000210855",
    "Acinetobacter baumannii AB5075-UW": "GCF_000963815",
    "Achromobacter xylosoxidans SOLR10": "GCF_003293615",
}

# dictionary for each species abbreviation and its full name
species_abb_to_name = {
    "ACHX": "Achromobacter xylosoxidans SOLR10",
    "ACIB": "Acinetobacter baumannii AB5075-UW",
    "AGGA": "Aggregatibacter actinomycetemcomitans D7S-1",
    "BBURG": "Borreliella burgdorferi B31",
    "BURK": "Burkholderia pseudomallei K96243",
    "Campy": "Campylobacter jejuni subsp. jejuni 81-176",
    "ENTFA": "Enterococcus faecalis OG1RF",
    "EPEC": "Escherichia coli EPEC 0127:H6 E2348/69",
    "ETEC": "Escherichia coli ETEC H10407",
    "UPEC": "Escherichia coli UPEC 536",
    "FRAT": "Francisella tularensis subsp. holarctica FSC200",
    "HINF": "Haemophilus influenzae 86-028NP",
    "HP_G27": "Helicobacter pylori G27",
    "HPG27": "Helicobacter pylori G27",  # duplicate w/o undetrscore
    "HPJ99": "Helicobacter pylori J99",
    "KLEBS": "Klebsiella pneumoniae subsp. pneumoniae MGH 78578",
    "LEGIP": "Legionella pneumophila subsp. pneumophila Philadelphia 1",
    "Listeria": "Listeria monocytogenes EGD-e",
    "MTB": "Mycobacterium tuberculosis H37Ra",
    "NGON": "Neisseria gonorrhoeae FA 1090",
    "NMEN": "Neisseria meningitidis serogroup C FAM18",
    "PSEUDO": "Pseudomonas aeruginosa PAO1",
    "SALMT": "Salmonella enterica subsp. enterica serovar Typhimurium SL1344",
    "SHIF": "Shigella flexneri 5a str. M90T",
    "MRSA": "Staphylococcus aureus MRSA252",
    "MSSA": "Staphylococcus aureus MSSA476",
    "SEPI": "Staphylococcus epidermidis 1457",
    "STAGA": "Streptococcus agalactiae NEM316",
    "STRPN": "Streptococcus pneumoniae D39",
    "SPYO": "Streptococcus pyogenes 5448",
    "SSUIS": "Streptococcus suis S10 -P1/7",
    "Vibrio": "Vibrio cholerae O1 biovar El Tor str. N16961",
    "YPSTB": "Yersinia pseudotuberculosis YPIII",
}

species_name_to_abb = {v: k for k, v in species_abb_to_name.items()}
species_name_to_abb["Borrelia burgdorferi B31"] = "BBURG"
species_name_to_abb["Helicobacter pylori G28"] = "HPG27"
species_name_to_abb["Staphylococcus aureus MRSA252"] = "MRSA"
species_name_to_abb["Staphylococcus aureus MSSA476"] = "MSSA"
species_name_to_abb["Staphylococcus epidermidis 1457"] = "SEPI"


stress_columns = [
    'As_1 (GE) - TPM', 'As_2 (GE) - TPM', 'As_3 (GE) - TPM',
    'Bs_1 (GE) - TPM', 'Bs_2 (GE) - TPM', 'Bs_3 (GE) - TPM',
    'Ctrl_1 (GE) - TPM', 'Ctrl_2 (GE) - TPM', 'Ctrl_3 (GE) - TPM',
    'Li_1 (GE) - TPM', 'Li_2 (GE) - TPM', 'Li_3 (GE) - TPM',
    'Mig_1 (GE) - TPM', 'Mig_2 (GE) - TPM', 'Mig_3 (GE) - TPM',
    'Nd_1 (GE) - TPM', 'Nd_2 (GE) - TPM', 'Nd_3 (GE) - TPM',
    'Ns_1 (GE) - TPM', 'Ns_2 (GE) - TPM', 'Ns_3 (GE) - TPM',
    'Oss_1 (GE) - TPM', 'Oss_2 (GE) - TPM', 'Oss_3 (GE) - TPM',
    'Oxs_1 (GE) - TPM', 'Oxs_2 (GE) - TPM', 'Oxs_3 (GE) - TPM',
    'Sp_1 (GE) - TPM', 'Sp_2 (GE) - TPM', 'Sp_3 (GE) - TPM',
    'Tm_1 (GE) - TPM', 'Tm_2 (GE) - TPM', 'Tm_3 (GE) - TPM',
    'Vic_1 (GE) - TPM', 'Vic_2 (GE) - TPM', 'Vic_3 (GE) - TPM'
]

def get_complement(sequence):
    return "".join([dna_complement[base] for base in sequence])


def get_reverse_complement(sequence):
    return get_complement(sequence[::-1])


def get_sequence_position(species_name, sequence, dna_sequences):
    # full_dna = dna_sequences[species_name]
    for key in dna_sequences[species_name].keys():
        full_dna = dna_sequences[species_name][key]
        if sequence in full_dna:
            return full_dna.find(sequence)
    return -1


def get_full_name(species_name):
    return species_abb_to_name[species_name]


def get_region_start(region_string):
    if "join" in region_string:
        return -5
    if "complement" in region_string:
        return int(region_string.replace("complement(", "").split("..")[0])
    return int(region_string.split("..")[0])


def get_region_end(region_string):
    if "join" in region_string:
        return -5
    if "complement" in region_string:
        return int(
            region_string.replace("complement(", "").replace(")", "").split("..")[1]
        )
    return int(region_string.split("..")[1])


class SequenceDataset(Dataset):
    """Dataset class with"""

    def __init__(self, X, y):
        # convert to float tensors
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X.iloc[idx]["species_1h"]).float(),
            torch.tensor(self.X.iloc[idx]["stress_name_1h"]).float(),
            torch.tensor(self.X.iloc[idx]["upstream200_1h"]).float().view(1, -1, 4),
        ), torch.tensor(self.y.iloc[idx]).float()
