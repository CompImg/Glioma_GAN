import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import nibabel as nib
import sys
import argparse

"""
ARG PARSING
"""
parser = argparse.ArgumentParser(description='mmGAN for glioma MRI image synthesis')
parser.add_argument('-t1',
                    dest='t1',
                    help='Path to T1 image (synthesized if non-existing)',
                    type=str,
                    required=True)
parser.add_argument('-t1c',
                    dest='t1c',
                    help='Path to T1ce image (synthesized if non-existing)',
                    type=str,
                    required=True)
parser.add_argument('-t2',
                    dest='t2',
                    help='Path to T2 image (synthesized if non-existing)',
                    type=str,
                    required=True)
parser.add_argument('-flair',
                    dest='flair',
                    help='Path to FLAIR image (synthesized if non-existing)',
                    type=str,
                    required=True)

args = parser.parse_args()
args_dict = vars(args)

"""
LOAD SEQUENCES (OR ADD EMPTY MASK FOR MISSING)
"""
sequences = ["t2","t1","t1c","flair"]
input_lst = []

for seq_ in sequences:
    if os.path.exists(args_dict[seq_]):
        img_file_ = nib.load(args_dict[seq_])
        img_ = img_file_.get_fdata()
        if img_.shape != (240,240,155):
            print(seq_ + " does not seem to be in SRI space; exiting!")
            exit()
        temp_bm = np.zeros(img_.shape)
        temp_bm[img_ != 0] = 1
        img_ = np.clip(img_,a_min=np.percentile(img_[temp_bm == 1],0.1),a_max=np.percentile(img_[temp_bm == 1],99.9))
        img_ -= img_[temp_bm==1].min()
        img_ /= img_[temp_bm==1].max()
        img_ *= temp_bm #Re-mask
        img_[(img_ == 0) & (temp_bm == 1)] = img_[img_ > 0].min() #Make sure 0's are only ever in the background
        img_ = img_[8:-8,8:-8,:] #Crop to (224,224,:)
        input_lst.append(img_)
    else: #For missing images, add empty mask
        input_lst.append(np.zeros((224,224,155)))

"""
IMAGE SYNTHESIS
"""
mdl = tf.keras.models.load_model("generator.h5", compile=False)

input_ = np.stack(input_lst,axis=-1)
input_ = np.expand_dims(input_,axis=0)

synth_imgs_lst_ = []
for x_axis_ctr_ in range(155):
    input_slice_ = input_[:,:,:,x_axis_ctr_,:]
    model_out_ = mdl.predict(input_slice_)
    synth_imgs_lst_.append(np.squeeze(model_out_))

synth_imgs_ = np.stack(synth_imgs_lst_,axis=-2) #Shape (224,224,155,4) -> Holds all four synthetic images in [sequences] order

for seq_ctr_ in range(len(sequences)):
    if not os.path.exists(args_dict[sequences[seq_ctr_]]):
        synth_img_ = np.squeeze(synth_imgs_[:,:,:,seq_ctr_])
        synth_img_ = np.pad(synth_img_,((8,8),(8,8),(0,0)))
        synth_img_ *= 1000
        synth_img_ *= temp_bm
        nib.save(nib.Nifti1Image(synth_img_.astype(np.float32),img_file_.affine,img_file_.header),args_dict[sequences[seq_ctr_]])
        print("Saved synthetic " + sequences[seq_ctr_] + " at: " + args_dict[sequences[seq_ctr_]])