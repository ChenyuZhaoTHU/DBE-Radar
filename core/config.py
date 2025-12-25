'''
Date: 2024-03-15
LastEditors: JJQ jj1623@ic.ac.uk
LastEditTime: 2024-07-15
FilePath: /coloradar/core/config.py
Description: 
'''
"""Configuration."""

import numpy as np

ROOTDIR: str = "dataset"

# Entry point of the dataset
DATASET: str = ROOTDIR + "/dataset.json"


# Minimum number of szimuth bins
# NUMBER_AZIMUTH_BINS_MIN: int = 32
NUMBER_AZIMUTH_BINS_MIN: int = 128

# Minimum number of elevation bins
# NUMBER_ELEVATION_BINS_MIN: int = 64
NUMBER_ELEVATION_BINS_MIN: int = 64

# Minimum number of doppler bins
# NUMBER_DOPPLER_BINS_MIN: int = 16
NUMBER_DOPPLER_BINS_MIN: int = 64

# Minimum number of range bins
NUMBER_RANGE_BINS_MIN: int = 128


# DoA estimation methods
# values: "fft", "esprit"
DOA_METHOD: str = "fft"

# Radar Digital Signal Processing Method
# values: "normal", "fesprit"
#   - "normal" stands for chain of Range-Doppler FFT, CFAR and DOA
#   - "fesprit" also computes the Range-Doppler FFT but uses ESPRIT for
#     a precise frequency estimation for Range, Doppler and DOA. No need
#     for CFAR
# NOTE: "fesprit" is still being tested and optimized
RDSP_METHOD: str = "normal"


# 2D Range-Doppler OS-CFAR Parameters used for generating
# radar pointclouds
RD_OS_CFAR_WS: int = 8         # Window size
RD_OS_CFAR_GS: int = 1         # Guard cell
RD_OS_CFAR_K: float = 0.75     # n'th quantile
RD_OS_CFAR_TOS: int = 8        # Tos factor

# 1D OS-CFAR Parameters used for peak selection in Azimuth-FFT
AZ_OS_CFAR_WS: int = 16         # Window size
AZ_OS_CFAR_GS: int = 8          # Guard cell
AZ_OS_CFAR_TOS: int = 6         # Tos factor


'''
Following are the parameters for the Beamforming algorithms
'''

# Beamforming method

BF_METHOD: str = "Capon"  # values: "Capon", "Music", "Esprit"

# parameters for Capon Beamforming
AZ_RANGE = 70
EL_RANGE = 25
ANGLE_RES_AZ: float = 0.75
ANGLE_RES_EL: float = 0.5
AZ_BINS = int(round(AZ_RANGE * 2 / ANGLE_RES_AZ))  + 1 # total number of azimuth bins
EL_BINS = int(round((EL_RANGE * 2 / ANGLE_RES_EL))) + 1 # total number of elevation bins

AZ_MESH = np.linspace(-AZ_RANGE, AZ_RANGE, AZ_BINS)* (np.pi / 180) # convert azimuth to radian
EL_MESH = np.linspace(-EL_RANGE, EL_RANGE, EL_BINS)* (np.pi / 180) # convert elevation to radian

# 2D Range-Azimuth OS-CFAR Parameters used for generating
# radar pointclouds
RA_OS_CFAR_WS: int = 10         # Window size
RA_OS_CFAR_GS: int = 8         # Guard cell
RA_OS_CFAR_K: float = 0.9     # n'th quantile
RA_OS_CFAR_TOS: int = 10        # Tos factor

# 1D OS-CFAR Parameters used for peak selection in Azimuth-FFT
EL_OS_CFAR_WS: int = 10         # Window size
EL_OS_CFAR_GS: int = 2          # Guard cell
EL_OS_CFAR_TOS: int = 2         # Tos factor

EL_CFAR_SKIP_ANGLE: int = 2
EL_CFAR_SKIP_BIN = int(round(EL_CFAR_SKIP_ANGLE / ANGLE_RES_EL))

# CFAR skip bin
CFAR_SKIP_RANGE_BIN_NEAR: int = 12 # equal to 1.4988686m for single chip radar
CFAR_SKIP_RANGE_BIN_FAR: int = 1 #  13 ~= 128* 0.1, remove 10% of the range bins
CFAR_SKIP_AZIMUTH_BIN: int = 4

