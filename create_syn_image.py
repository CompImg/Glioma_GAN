import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import nibabel as nib
import sys

sequences = ["t2","t1","t1c","flair"]
seq_dict = {}

#Parse arguments
if len(sys.argv) != 4:
    print("Invalid number of arguments. This script requires paths to three of the four BraTS sequences to synthesize the fourth sequence:")
    print("--t2=/path/to/t2.nii.gz")
    print("--t1=/path/to/t1.nii.gz")
    print("--t1c=/path/to/t1c.nii.gz")
    print("--flair=/path/to/flair.nii.gz")
    exit()

for elem_ in sys.argv[1:]:
    for seq_ in sequences:
        if "--"+seq_+"=" in elem_:
            if not os.path.exists(elem_[elem_.find("=")+1:]):
                print(seq_ + " file not found; check path!")
                exit()
            else:
                img_file = nib.load(elem_[elem_.find("=")+1:])
                seq_dict[seq_] = img_file.get_fdata()
                if img_file.shape != (240,240,155):
                    print(seq_ + " does not seem to be in SRI space; exiting")
                    exit()

if len(seq_dict) != 3:
    print("Invalid arguments. This script requires paths (no spaces!) to three of the four BraTS sequences to synthesize the fourth sequence:")
    print("--t2=/path/to/t2.nii.gz")
    print("--t1=/path/to/t1.nii.gz")
    print("--t1c=/path/to/t1c.nii.gz")
    print("--flair=/path/to/flair.nii.gz")
    exit()

mdl = tf.keras.models.load_model("generator.h5", compile=False)

for key_ in seq_dict.keys():
    temp_bm = np.zeros(seq_dict[key_].shape)
    temp_bm[seq_dict[key_] != 0] = 1

    seq_dict[key_] = np.clip(seq_dict[key_],a_min=np.percentile(seq_dict[key_][temp_bm == 1],0.1),a_max=np.percentile(seq_dict[key_][temp_bm == 1],99.9))
    seq_dict[key_] -= seq_dict[key_][temp_bm==1].min()
    seq_dict[key_] /= seq_dict[key_][temp_bm==1].max()
    seq_dict[key_] *= temp_bm #Re-mask
    seq_dict[key_][(seq_dict[key_] == 0) & (temp_bm == 1)] = seq_dict[key_][seq_dict[key_] > 0].min()
    seq_dict[key_] = seq_dict[key_][8:-8,8:-8,:]

input_lst = []
for seq_ctr_ in range(len(sequences)):
    if sequences[seq_ctr_] in seq_dict.keys():
        input_lst.append(seq_dict[sequences[seq_ctr_]])
    else:
        input_lst.append(np.zeros((224,224,155)))
        missing_seq = seq_ctr_

input_ = np.stack(input_lst,axis=-1)
input_ = np.expand_dims(input_,axis=0)

synth_seq_lst = []
for x_axis_ctr_ in range(155):
    input_slice_ = input_[:,:,:,x_axis_ctr_,:]
    model_out_ = mdl.predict(input_slice_)
    synth_seq_lst.append(np.squeeze(model_out_[:,:,:,missing_seq]))

synth_img = np.stack(synth_seq_lst,axis=-1)
synth_img = np.pad(synth_img,((8,8),(8,8),(0,0)))
synth_img *= 1000
synth_img *= temp_bm

new_path = sys.argv[1][sys.argv[1].find("=")+1:]
new_path = os.path.dirname(new_path) + "/synth_" + sequences[missing_seq] + ".nii.gz"
nib.save(nib.Nifti1Image(synth_img.astype(np.float32),img_file.affine,img_file.header),new_path)

print("Synthetic " + sequences[missing_seq] + " created and saved at " + new_path)