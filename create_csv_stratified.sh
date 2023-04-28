#!/bin/bash

# print the title line in the csv file

echo "file_name,patient_id,label" > training_bm.csv && \
for y in {benign,cancer}; \
do for z in $(ls ${y}); \
do \
name=$(echo ${z} | cut -d '_' -f1) && \
echo ${y}/${z},${name},${y} >> training_bm.csv; \
done; done