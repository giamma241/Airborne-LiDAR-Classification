"""
Wrapper to apply Cloth Simulation Filtering using cloth-simulation-filter
"""

from csf import CSF


def apply_csf(xyz_array):
    csf = CSF()
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 0.5
    csf.params.rigidness = 3
    ground, non_ground = csf.do_filtering(xyz_array)
    return ground, non_ground
