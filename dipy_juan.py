# %%
import os
import bids
import numpy as np
import nibabel as nib
import subprocess
import shutil
import matplotlib.pyplot as plt
import dipy.reconst.shm as shm
import dipy.direction.peaks as dp
import matplotlib.pyplot as plt

from bids import BIDSLayout
from bids.tests import get_test_data_path
from dipy.io.image import save_nifti
from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.mcsd import (auto_response_msmt, mask_for_response_msmt, response_from_mask_msmt)
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.viz import window, actor
from dipy.data import get_sphere, get_fnames
from dipy.data import get_fnames, small_sphere, default_sphere
from dipy.direction import ProbabilisticDirectionGetter, peaks_from_model
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.dki import DiffusionKurtosisModel
from dipy.direction import PTTDirectionGetter
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.viz import window, actor, colormap, has_fury

# %%
sphere = get_sphere('symmetric724')
interactive = False
global_path = '/opt/dora/Dora/BIDS_ELA/BIDS_ELA/'
data_path = os.path.join(get_test_data_path(), global_path)
print(data_path)
layout = BIDSLayout(data_path)
subjs = layout.get_subjects()
print(subjs)

# %%
subjs = ['C001','C002','C003','C006','C007', 'P002','P003','P006','P007','P009']
# subjs = ['C001']

# %%
# for subj in subjs:
#     print(subj)
#     out_dir = global_path +'/derivatives/ELA_dipy/sub-' + subj
#     if subj.startswith('C'):
#          out_dir_anatomical = global_path +'/derivatives/ELA_dipy/dipy_juan/control/sub-' + subj + '/anatomical_measures'
#     elif subj.startswith('P'):
#        out_dir_anatomical = global_path +'/derivatives/ELA_dipy/dipy_juan/patient/sub-' + subj + '/anatomical_measures'
#     else:
#         print(f"{subj} no es un código válido")
#     if not os.path.exists(out_dir_anatomical):
#         os.makedirs(out_dir_anatomical)
    
   
#     data_path = out_dir + '/sub-' + subj+ '_mc_eddy_den_unring_images.nii.gz'
#     bval_path = global_path + '/sub-' + subj + '/dwi/sub-' + subj + '_run-1_dwi.bval'
#     bvec_path =  out_dir +  '/eddy/eddy_unwarped_images.eddy_rotated_bvecs'
#     mask_path = out_dir + '/sub-' + subj + '_brainmask.nii.gz'

#     #RECONSTRUCCION CSD
#     cmd = 'dipy_fit_csd ' + data_path + ' ' + bval_path + ' ' + bvec_path + ' ' + mask_path + ' --fa_thr 0.7 --sh_order 8 --parallel  --out_dir ' + out_dir_anatomical + ' --out_pam csd_peaks.pam5 --out_shm csd_shm.nii.gz --out_peaks_dir csd_peaks_dirs.nii.gz --out_peaks_values csd_peaks_values.nii.gz --out_peaks_indices csd_peaks_indices.nii.gz --out_gfa csd_gfa.nii.gz --force'
#     print (cmd)
#     subprocess.run(cmd, shell = True)

    

#     #RECONSTRUCCION CSA
#     cmd = (
#         'dipy_fit_csa ' + data_path + ' ' + bval_path + ' ' + bvec_path + ' ' + mask_path + 
#         ' --out_dir ' + out_dir_anatomical + 
#         ' --out_pam csa_peaks.pam5' +
#         ' --force'
#     )
#     print (cmd)
#     subprocess.run(cmd, shell = True)
    
#     #RECONSTRUCCION DTI
  
#     cmd = (
#         'dipy_fit_dti ' + data_path + ' ' + bval_path + ' ' + bvec_path + ' ' + mask_path + 
#         ' --save_metrics "md" "fa" "ad" "rd"' + 
#         ' --out_dir ' + out_dir_anatomical + 
#         ' --out_fa dti_fa.nii.gz' +
#         ' --out_ad dti_ad.nii.gz' + 
#         ' --out_rd dti_rd.nii.gz' +
#         ' --out_md dti_md.nii.gz'+
#         ' --force')
#     print (cmd)
#     subprocess.run(cmd, shell = True)
   
#    #RECONSTRUCCION DKI

#     cmd = 'dipy_fit_dki ' + data_path + ' ' + bval_path + ' ' + bvec_path + ' ' + mask_path + ' --b0_threshold 70.0 --save_metrics "md" "fa"  "ad" "rd" --out_dir ' + out_dir_anatomical + ' --out_fa dki_fa.nii.gz  --out_ad dki_ad.nii.gz --out_rd dki_rd.nii.gz --out_md dki_md.nii.gz --force'
#     print (cmd)
#     subprocess.run(cmd, shell = True)
# %%


for subj in subjs:
    print(subj)
    atlas = '/home/alumno/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk'
    in_trk_dir = global_path + "/derivatives/ELA_dipy/sub-" + subj + "/tracking/probabilistic/csd/tractogram_probabilistic_dg_pmf.trk"
    out_dir = global_path + "/derivatives/ELA_dipy/sub-" + subj + "/tracking/probabilistic/csd/"

    cmd = 'dipy_slr ' + atlas + ' ' + in_trk_dir + ' --out_dir ' + out_dir + ' --force'
    print(cmd)
    # os.system(cmd)

# %% 

for subj in subjs:
    if subj.startswith('C'):
        out_dir_rec = global_path + '/derivatives/ELA_dipy/dipy_juan/control/sub-' + subj + '/rec_bundles'
    elif subj.startswith('P'):
        out_dir_rec = global_path + '/derivatives/ELA_dipy/dipy_juan/patient/sub-' + subj + '/rec_bundles'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    in_dir = global_path + '/derivatives/ELA_dipy/sub-' + subj + '/tracking/probabilistic/csd/moved.trk'
    bundles_dir = '/opt/dora/Dora/BIDS_ELA/BIDS_ELA/derivatives/ELA_dipy/bundles/*.trk'
    cmd = 'dipy_recobundles "' + in_dir + '" "' + bundles_dir + '" --out_dir ' + out_dir_rec + ' --force --mix_names'
    print(cmd)
    os.system(cmd)

    if subj.startswith('C'):
        out_dir_org = global_path + '/derivatives/ELA_dipy/dipy_juan/control/sub-' + subj + '/org_bundles'
    elif subj.startswith('P'):
        out_dir_org = global_path + '/derivatives/ELA_dipy/dipy_juan/patient/sub-' + subj + '/org_bundles'

    if not os.path.exists(out_dir_org):
        os.makedirs(out_dir_org)

    
    out_dir_rec = out_dir_rec + '/*.npy'
    cmd = "dipy_labelsbundles " + in_trk_dir + " '" + out_dir_rec + "' --mix_names --out_dir " + out_dir_org + " --force"
    print(cmd)
    os.system(cmd)




