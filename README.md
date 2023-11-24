# Glioma GAN

Scripts and model accompanying our paper ["Improving Automated Glioma Segmentation in Routine Clinical Use Through Artificial Intelligence-Based Replacement of Missing Sequences With Synthetic Magnetic Resonance Imaging Scans"](https://pubmed.ncbi.nlm.nih.gov/34652289/) (Invest Radiol. 2022 Mar 1;57(3):187-193.).

The program can synthesize missing MR modalities to enable downstream processing of glioma examinations.

##  Usage and file description
**generator.h5**: The fully trained model used for evaluation in our manuscript.

**create_syn_image.py**: This script will create missing MR sequences (T1, T1ce, T2 or FLAIR) from available input modalities. This script accepts as input parameters the file paths to all four "BraTS" sequences:
> -t2 /path/to/t2.nii.gz

> -t1 /path/to/t1.nii.gz

> -t1c /path/to/t1c.nii.gz

> -flair /path/to/flair.nii.gz

All missing modalities are synthesized and stored in the path given above. Please ensure that all input images are correctly preprocessed (like in the example; you can use [BraTS.Toolkit](https://github.com/neuronflow/BraTS-Toolkit) for that). The file "generator.h5" must be in the same directory as the script. To for example create a synthetic flair from the example (assuming the files are stored in data/) call:

*python3 create_syn_image.py -t1 data/example_t1.nii.gz -t1c data/example_t1c.nii.gz -t2 data/example_t2.nii.gz -flair data/example_flair_synth.nii.gz*

**/data/**: Four example NIfTIs, already fully preprocessed with [BraTS.Toolkit](https://pubmed.ncbi.nlm.nih.gov/32410929/).

**/training/mmgan_brats.py**: Code used for training the GAN.


## Expected input data
The program works best with co-registered and skullstripped files in SRI-24 space. These can be obtained from [BrainLes](https://github.com/BrainLesion/preprocessing) or [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit)

## Citation
If you use this program, please cite to support our development:

Thomas, M. F., Kofler, F., Grundl, L., Finck, T., Li, H., Zimmer, C., Menze, B., & Wiestler, B. (2022). Improving Automated Glioma Segmentation in Routine Clinical Use Through Artificial Intelligence-Based Replacement of Missing Sequences With Synthetic Magnetic Resonance Imaging Scans. Investigative radiology, 57(3), 187–193. https://doi.org/10.1097/RLI.0000000000000828
```
@article{thomas2022improving,
  title={Improving automated glioma segmentation in routine clinical use through artificial intelligence-based replacement of missing sequences with synthetic magnetic resonance imaging scans},
  author={Thomas, Marie Franziska and Kofler, Florian and Grundl, Lioba and Finck, Tom and Li, Hongwei and Zimmer, Claus and Menze, Bj{\"o}rn and Wiestler, Benedikt},
  journal={Investigative Radiology},
  volume={57},
  number={3},
  pages={187--193},
  year={2022},
  publisher={LWW}
}
```
