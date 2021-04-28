# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import nibabel as nib
import gc
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from scipy.ndimage import shift, rotate

#hyperparameters
sequences = ["t2","t1","t1c","flair"]
gf = 32 #Number of filters in the generator
df = 32 #Number of filters in the discriminator
channel_noise = True #Add random gamma and gaussian smoothing to each input channel individually
gan_lambda = 10. # lambda term in the combined loss: BCE + gan_lambda*atten_loss; higher lambda favors img_loss over BCE
img_lambda = 1. # Lambda term in the AttentionLoss: SSIM(img) + mae_lambda*L1(lesions); higher lambda favors attention_loss
disc_update_decay = 0.5 # Multiplied with the discriminator loss to slow it down
loss_fct = "ssim" # Either "ssim" or "l1"; defines the loss for the whole image
augmentation_probability = 0.5 # Augment image(s) if random.random() < augmentation_probability
img_shape = (224,224,4)
batch_size = 1
epochs = 151
lr = 0.001 # Learning rate for the Adam optimizer

folder_name = "mmGAN_bs-" + str(batch_size) + "_GanL-" + str(gan_lambda) + "_ImgL-" + str(img_lambda) + "_DiscUpdateDecay-" + str(disc_update_decay) + "_loss-" + loss_fct + "_LR-" + str(lr) + "_ChannelNoise-" + str(channel_noise)
if not os.path.exists("/mnt/Drive1/bene/brats_gan/" + folder_name + "/"):
  os.mkdir("/mnt/Drive1/bene/brats_gan/" + folder_name + "/")

def build_generator(gf=32):
  """U-Net Generator"""

  def conv2d(layer_input, filters, downsample = True):
    """Layers used during downsampling"""

    if downsample:
      layer_input = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)
      layer_input = tfa.layers.InstanceNormalization() (layer_input)
      layer_input = tf.keras.layers.LeakyReLU()(layer_input)
    
    d = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)
    d = tfa.layers.InstanceNormalization() (d)
    d = tf.keras.layers.LeakyReLU()(d)

    d = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(d)
    d = tfa.layers.InstanceNormalization() (d)
    d = tf.keras.layers.LeakyReLU()(d)

    return d

  def deconv2d(layer_input, skip_input, filters):
    """Layers used during upsampling"""

    u = tf.keras.layers.Conv2DTranspose(filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)
    u = tfa.layers.InstanceNormalization() (u)
    u = tf.keras.layers.LeakyReLU()(u)

    u = tf.keras.layers.Concatenate()([u, skip_input])

    u = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(u)
    u = tfa.layers.InstanceNormalization() (u)
    u = tf.keras.layers.LeakyReLU()(u)

    u = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(u)
    u = tfa.layers.InstanceNormalization() (u)
    u = tf.keras.layers.LeakyReLU()(u)

    return u

  # Image input
  input_img = tf.keras.layers.Input(shape=img_shape,name="d0")
  d0 = conv2d(input_img, gf*1,downsample=False)

  # Downsampling
  d1 = conv2d(d0, gf*2)
  d2 = conv2d(d1, gf*3)
  d3 = conv2d(d2, gf*4)
  d4 = conv2d(d3, gf*5)
        
  # Bottleneck
  d5 = conv2d(d4, gf*5)

  # Upsampling
  u4 = deconv2d(d5, d4, gf*4)
  u3 = deconv2d(u4, d3, gf*3)
  u2 = deconv2d(u3, d2, gf*2)
  u1 = deconv2d(u2, d1, gf*1)

  # Image generation @ full resolution
  u0 = tf.keras.layers.Conv2DTranspose(gf*1, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False)(u1)
  u0 = tfa.layers.InstanceNormalization() (u0)
  u0 = tf.keras.layers.LeakyReLU()(u0)

  u0 = tf.keras.layers.Concatenate()([u0, d0])

  u0 = tf.keras.layers.Conv2D(gf*1, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(u0)
  u0 = tfa.layers.InstanceNormalization() (u0)
  u0 = tf.keras.layers.LeakyReLU()(u0)
  
  u0 = tf.keras.layers.Conv2D(gf*1, kernel_size=3, strides=1, kernel_initializer='he_uniform', padding='same', use_bias=False)(u0)
  u0 = tfa.layers.InstanceNormalization() (u0)
  u0 = tf.keras.layers.LeakyReLU()(u0)
  
  output_img = tf.keras.layers.Conv2D(img_shape[2], kernel_size=1, strides=1, padding='same', activation='relu')(u0)

  return tf.keras.Model(input_img, output_img)

def build_discriminator(df=32):
  """The discriminator (ImageGAN)"""
  #https://fomoro.com/research/article/receptive-field-calculator

  def d_layer(layer_input, filters):
    """Discriminator layer"""
    
    d = tf.keras.layers.Conv2D(filters, kernel_size=4, strides=2, kernel_initializer='he_uniform', padding='same', use_bias=False)(layer_input)
    d = tfa.layers.InstanceNormalization() (d)
    d = tf.keras.layers.LeakyReLU()(d)

    return d

  input_img = tf.keras.layers.Input(shape=img_shape)
  d1 = d_layer(input_img, df)
  d2 = d_layer(d1, df*2)
  d3 = d_layer(d2, df*4)
  d4 = d_layer(d3, df*6)
  d5 = d_layer(d4, df*8)

  validity = tf.keras.layers.Conv2D(1, kernel_size=1, strides=1, padding='same', activation="linear")(d5)

  return tf.keras.Model(input_img, validity)

discriminator = build_discriminator(df)
generator = build_generator(gf)

loss_object = tf.keras.losses.MeanSquaredError() #For the discriminator in a ls gan

def generator_loss(disc_generated_output, gen_output, target, gt):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # recon error, w/ attention loss
  ssim_loss_img = 1. - tf.reduce_min(tf.image.ssim(target, gen_output, max_val=1)) #We input all four sequences, but three are all zeros (i.e. SSIM == 1), so only the minimum SSIM is the "true loss"
  l1_loss_img = tf.reduce_mean(tf.abs(target - gen_output))
  abs_diff_seg = tf.reduce_sum( tf.abs(tf.math.multiply(target,gt) - tf.math.multiply(gen_output,gt) ) )
  l1_loss_seg = tf.math.divide(abs_diff_seg,tf.math.reduce_sum(gt)+0.00000001) 
  if loss_fct == "ssim":
    img_loss = ssim_loss_img + (img_lambda * l1_loss_seg)
  if loss_fct == "l1":
    img_loss = l1_loss_img + (img_lambda * l1_loss_seg)

  total_gen_loss = gan_loss + (gan_lambda * img_loss)

  return total_gen_loss, gan_loss, img_loss

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss

  return disc_update_decay * total_disc_loss

# A couple of helper functions
def crop_to_shape(img_arr):
  """Crops input image array to target shape"""
  difference_0 = img_arr.shape[0] - img_shape[0]
  difference_1 = img_arr.shape[1] - img_shape[1]
  img_arr_cropped = img_arr[(difference_0 // 2):img_arr.shape[0] - (difference_0 // 2),(difference_1 // 2):img_arr.shape[1] - (difference_1 // 2)]

  return img_arr_cropped

def yield_batch(ids,training=True):
  """ids is a list of len(batch_size) with file names"""
  imgs_source = []
  masks_source = []
  masks_target = []
  imgs_gt = []
  for img_path in ids:
    target_sequence = random.randint(0,3) #Randomly sample an index [0..3] of the target sequence for this example
    src_tmp_lst = []
    for seq in sequences:
      img_src_pil = Image.open(img_path.replace('flair',seq))
      img_src = np.asarray(img_src_pil,dtype=np.float32)
      img_src = crop_to_shape(img_src)
      img_src /= 255.
      if (channel_noise) and (random.random() < 0.33) and (training): #We add noise channel-wise, but only during training
        gamma = random.uniform(0.5, 1.5)
        img_src = adjust_gamma(img_src, gamma=gamma)
      if (channel_noise) and (random.random() < 0.33) and (training):
        sigma = random.uniform(0.5, 1.5)
        img_src = gaussian(img_src, sigma=sigma)
      src_tmp_lst.append(img_src)

    img_gt_pil = Image.open(img_path.replace("flair","gt"))
    img_gt = np.asarray(img_gt_pil,dtype=np.float32)
    img_gt = crop_to_shape(img_gt)

    img_src = np.stack(src_tmp_lst,axis=-1)
    img_gt = np.expand_dims(img_gt,axis=-1)

    if (np.random.random() < augmentation_probability) and (training):
      img_src = np.fliplr(img_src)
      img_gt = np.fliplr(img_gt)

    if (np.random.random() < augmentation_probability) and (training):
      img_src = np.flipud(img_src)
      img_gt = np.flipud(img_gt)

    if (np.random.random() < augmentation_probability) and (training):
      rot_angle = random.randint(-45, 45)
      img_src = rotate(img_src, rot_angle, reshape=False, order=1)
      img_gt = rotate(img_gt, rot_angle, reshape=False, order=0)

    if (np.random.random() < augmentation_probability) and (training):
      x_shift = random.randint(-45, 45)
      y_shift = random.randint(-45, 45)
      img_src = shift(img_src, [x_shift, y_shift, 0],order=1)
      img_gt = shift(img_gt, [x_shift, y_shift, 0],order=0)

    mask_source = np.ones(img_src.shape)
    mask_source[:,:,target_sequence] = 0

    mask_target = np.zeros(img_src.shape)
    mask_target[:,:,target_sequence] = 1

    imgs_source.append(img_src)
    imgs_gt.append(img_gt)
    masks_source.append(mask_source)
    masks_target.append(mask_target)

  imgs_source = np.array(imgs_source,dtype=np.float32)
  imgs_gt = np.array(imgs_gt,dtype=np.float32)
  masks_source = np.array(masks_source,dtype=np.float32)
  masks_target = np.array(masks_target,dtype=np.float32)

  return imgs_source,imgs_gt,masks_source,masks_target

def sample_images(epoch,imgs_src,model):
  for ctr in range(len(sequences)):
    imgs_src_tmp = np.copy(imgs_src)
    imgs_src_tmp[:,:,:,ctr] = np.zeros(imgs_src_tmp[:,:,:,ctr].shape)
    fake_img = model(imgs_src_tmp, training=False)
    for img_ctr in range(imgs_src.shape[0]):
      dpi = 96
      w = 5*imgs_src.shape[1]/dpi
      h = 3*imgs_src.shape[2]/dpi
      plt.figure(figsize=(w, h))
      plt.subplot(1,2,1).set_title("Generated " + sequences[ctr])
      plt.imshow(fake_img[img_ctr,:,:,ctr],cmap="gray")
      plt.subplot(1,2,2).set_title("Target " + sequences[ctr])
      plt.imshow(imgs_src[img_ctr,:,:,ctr],cmap="gray")
      plt.savefig("/mnt/Drive1/bene/brats_gan/" + folder_name + "/epoch%d_img%d_%s.png" % (epoch, img_ctr, sequences[ctr]), dpi=dpi)
      plt.close()

generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

@tf.function
def train_step(src, gt, src_masks, tar_masks, epoch):
  #src_masks = All ones except for the target sequence
  #tar_masks = All zeros except for the target sequence
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(tf.multiply(src,src_masks), training=True)

    disc_real_output = discriminator(src, training=True)
    disc_generated_output = discriminator(tf.add(tf.multiply(gen_output,tar_masks),tf.multiply(src,src_masks)), training=True) #Just replace target with the generator output

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, tf.multiply(gen_output,tar_masks), tf.multiply(src,tar_masks), gt)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

def fit_gan():
  path = glob('/mnt/Drive1/bene/brats_gan/pngs/train/flair/*')
  path_valid = glob('/mnt/Drive1/bene/brats_gan/pngs/valid/flair/*')
  bat_per_epo = int(len(path) / batch_size)
  for epoch in range(epochs):
    random.shuffle(path)
    for batch in range(bat_per_epo):
      ids = path[(batch*batch_size):((batch+1)*batch_size)]
      imgs_src,imgs_gt,masks_source,masks_target = yield_batch(ids,training=True)
      _, gen_gan_loss, gen_img_loss, disc_loss = train_step(imgs_src, imgs_gt, masks_source, masks_target, epoch)
      print("Batch %d/%d / Epoch %d/%d: Gen_Disc_loss [%.3f], Gen_Img_loss [%.3f], Disc_loss [%.3f]" % (batch+1, bat_per_epo, epoch+1, epochs, np.around(gen_gan_loss,3), np.around(gen_img_loss,3),np.around(disc_loss,3)))
 
    #Sample 3 images from validation at epoch's end and plot them
    random.shuffle(path_valid)
    ids_valid = path_valid[0:3]
    imgs_src_valid,_,_,_ = yield_batch(ids_valid,training=False)
    sample_images(epoch+1,imgs_src_valid,generator)
    if (epoch+1) % 15 == 0:
      generator.save("/mnt/Drive1/bene/brats_gan/" + folder_name + "/generator_epoch%d.h5" % (epoch+1))
    gc.collect() #Garbage collection; save some RAM

fit_gan()
generator.save("/mnt/Drive1/bene/brats_gan/" + folder_name + "/generator_final_epoch%d.h5" % (epoch+1))