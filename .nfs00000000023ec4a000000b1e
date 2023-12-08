#!/bin/bash
# download and process data

mkdir -p data

while getopts ":c" opt; do
    case $opt in
        c)
            echo "Option -c selected, which will clear the datasets folder and re-process all data files."
            read -r -p "Are you sure you want to proceed? [y/n]" prompt
            if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]
            then
                rm -r data/datasets/*
            fi
    esac
done

echo "downloading CITE-seq data if needed..."

CITEDATA=data/raw/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz

if ! [[ -f "$CITEDATA" ]]; then
    wget -q -P data/raw/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed.h5ad.gz
    gzip -d data/raw/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
fi

cd data/scripts/

echo "Processing CITE-seq data if needed..."

python process_cite.py

echo "Creating interventional balls data if needed..."

python generate_balls.py --distribution_case intervention --scm_mechanism non_linear --latent_case scm


