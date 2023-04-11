#!/bin/bash

yes | conda install pandas && \
yes | conda install scipy && \
yes | conda install -c conda-forge openjpeg jpeg && \
yes | conda install pillow && \
yes | conda install -c conda-forge jupyterlab && \
yes | pip install python-gdcm && \
yes | conda install -c conda-forge pydicom && \
yes | conda install matplotlib