#!/bin/bash

# Decompress mutants2 files
mkdir -p /scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs/mutants
for file in /scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_compressed/mutants/*.tar.gz; do
    tar -xzf "$file" -C /scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs/mutants/
    echo "Decompressed $file"
done

# Decompress wild_type files
mkdir -p /scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs/wild_type
for file in /scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_compressed/wild_type/*.tar.gz; do
    tar -xzf "$file" -C /scratch/leuven/359/vsc35907/big_data_feature_extraction/patches_dirs/wild_type/
    echo "Decompressed $file"
done