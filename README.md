
# Forecasting Pathogenic Strain Responses Under Diverse Stress Conditions

This project aims to predict expression profiles of pathogenic bacteria under stress by leveraging sequence-based models. Despite the simplicity of prokaryotic genomes, which consist of a single circular chromosome organized into operons, they have not been as extensively studied as eukaryotic genomes. Recent advancements have shown that language models can identify elements of prokaryotic genomes, prompting us to explore their potential for predicting gene expression, similar to efforts in yeast. Our objectives include benchmarking various model architectures using mean squared error, interpreting models to identify predictive regulatory elements, assessing the specificity of these elements under different stress conditions, and identifying co-regulated transcripts. By doing so, we aim to enhance our understanding of bacterial gene regulation, providing insights with potential applications in biotechnology and industries such as food production.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Updating the Environment](#updating-the-environment)
- [Building the Data](#building-the-data)

## Prerequisites

Ensure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

## Setting Up the Development Environment

To create the conda environment and install all necessary packages, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create the conda environment from the `environment.yml` file:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate ml4rg
   ```

## Updating the Environment

When the environment needs updates (e.g., adding new packages), update your local environment and re-export the `environment.yml` file as follows:

1. **Install the new package:**
   ```bash
   conda install some_new_package
   ```

2. **Export the updated environment:**
   ```bash
   conda env export --name ml4rg > environment.yml
   ```

3. **Commit and push the updated `environment.yml` file to the repository:**
   ```bash
   git add environment.yml
   git commit -m "Update environment.yml with new packages"
   git push origin main
   ```

## Building the Data

Run the script `sync_drive.sh` to build the data:
```bash
bash sync_drive.sh
```
### Data Sequences

**.fna Files**
These are FASTA files containing nucleotide sequences. Each sequence in the file starts with a header line beginning with >, followed by lines of nucleotide sequences.

**.gff Files**
These are General Feature Format files, which contain information about gene features like gene locations, exons, introns, etc. Each line in a GFF file represents one feature with fields separated by tabs.

### Formatter Status

![Auto formate code](https://github.com/memetc/ML4RG_Group1/workflows/Auto%20formate%20code/badge.svg)

## Steps to Prepare and Analyze Data

### 1. Organize Data Directory
- Create a directory named `data`.
- Move the folders `data_expression` and `data_sequences_upstream` into the `data` directory.

### 2. Merge Data
- Run `load_data.py` to merge the expression data with sequence data.
- The resulting merged data is saved as `merged_data.csv` inside the `data` directory.

### 3. Process Data
- Run `process_data.py` to process the merged data.
- The processed data is saved as `processed_data.pkl` in the `data` directory.

### 4. Model Development
- Open the `model_development` notebook located in the `notebooks` directory.
- Train the model and review the results by following the instructions in the notebook.
