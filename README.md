# Glioma_GAN

Scripts and model accompanying our paper "Improving automated glioma segmentation in routine clinical use through AI-based replacement of missing sequences with synthetic MR images".

**generator.h5**: The fully trained model also used for evaluation in our manuscript.

**mmgan_brats.py**: The code used for defining and training the model from TCIA data.

**example_niftis.zip**: A single example of a glioma patient with all for modalities present and correctly preprocessed (i.e. registered and resampled to SRI space and skullstripped).

**create_syn_image.py**: This script will create a missing MR sequence (T1, T1ce, T2 or FLAIR) from the three other modalities. This script accepts as input parameters the file paths to the three input modalities like:
> --t2=/path/to/t2.nii.gz

> --t1=/path/to/t1.nii.gz

> --t1c=/path/to/t1c.nii.gz

> --flair=/path/to/flair.nii.gz

The fourth (missing) sequence is synthesized. Please ensure that all input images are correctly preprocessed (like in the example; you can use [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit) for that) and omit spaces in the file paths. The file "generator.h5" must be in the same directory as the script. To for example create a synthetic flair from the example (assuming the files are stored in /home/you/) call:

*python3 create_syn_image.py --t2=/home/you/example_t2.nii.gz --t1c=/home/you/example_t1c.nii.gz --t1=/home/you/example_t1.nii.gz*
