import os
import logging
import time
import sys

SINE_MODEL_PATH_DIC = {
    'epinions': './embeddings/sine_epinions_models',
    'slashdot': './embeddings/sine_slashdot_models',
    'bitcoin_alpha': './embeddings/sine_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/sine_bitcoin_otc_models'
}

SIDE_MODEL_PATH_DIC = {
    'epinions': './embeddings/side_epinions_models',
    'slashdot': './embeddings/side_slashdot_models',
    'bitcoin_alpha': './embeddings/side_bitcoin_alpha_models',
    'bitcoin_otc': './embeddings/side_bitcoin_otc_models'
}

DATASET_NUM_DIC = {
    'epinions': 131828,
    'slashdot': 82140,
    'bitcoin_alpha': 3783,
    'bitcoin_otc': 5881,
}

EMBEDDING_SIZE = 20