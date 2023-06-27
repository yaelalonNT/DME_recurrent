# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:17:04 2023

@author: yael
"""
import pandas as pd
import os
import numpy as np
import imutils
import random
from scipy.ndimage import rotate

class HoctDataset():
    def __init__(self,args,split):
        self.data_csv_name = 'data_summary.csv'
        self.length_clip = args.length_clip # Sequence length for training
        self.split = split 
        self.num_labels = args.maxseqlen
        self.data_dir = args.hoct_dir
        self.augment_prob_xflip = args.augment_prob_xflip
        self.augment_prob_yflip = args.augment_prob_yflip
        self.augment_prob_rotate = args.augment_prob_rotate
        self.numpy_dir = os.path.join(self.data_dir,'numpy')
        self.data_df = pd.read_csv(os.path.join(self.data_dir,self.data_csv_name))
        self.get_sequences()

    def get_sequences(self):
        """
        Creates three lists - self.sequences_images, which includes the full path of the images 
        of each sequence, self.sequences_annotations - the full path of segmentation data 
        per sequence, and self.seq_names - list of sequences name
        """
        if self.split == 'train':
            self.df_dataset = self.data_df[self.data_df['data set']=='train']
        else:
            self.df_dataset = self.data_df[self.data_df['data set']=='validation']            
        
        self.sequences_images = []
        self.sequences_annotations = []
        self.seq_names = []
        for scan in self.df_dataset.Scan:
            scan = scan.replace(' ','')   
            scan_path_img = os.path.join(self.numpy_dir,'I_VS',scan)
            scan_path_seg = os.path.join(self.numpy_dir,'Seg',scan)
            dir_files = os.listdir(scan_path_img)
            dir_numbers = sorted([int(dir_files[i].replace('.npy','')) for i in range(0,len(dir_files))])
            for n in range(0,len(dir_numbers),self.length_clip):
                if n + self.length_clip < len(dir_files): 
                    ind = range(n,n+self.length_clip) 
                else: # End of volume scan
                    min_ind = max(0,len(dir_files)-self.length_clip)
                    ind = range(min_ind,len(dir_files))
                    
                seq_img_path = [os.path.join(scan_path_img,str(dir_numbers[f]) + '.npy') for f in ind]
                seq_seg_path = [os.path.join(scan_path_seg,str(dir_numbers[f]) + '.npy') for f in ind]
                seq_name = '%s_%d_%d' %(scan,dir_numbers[ind[0]],dir_numbers[ind[-1]])
                self.sequences_images.append(seq_img_path)
                self.sequences_annotations.append(seq_seg_path)
                self.seq_names.append(seq_name)
        return
    
    def rotate_image_nearest(self,image, angle):
        """
        Rotate image using nearst neighbor method
        """
        image_copy = image.copy()
        rotated_image = rotate(image_copy, angle, reshape=False, order=0)
        return rotated_image
    
    def rotate_target(self,target,angle):
        """
        Rotate target using the specified angle 
        """
        target_labels = np.argmax(target,axis=0)
        target_rotate = self.rotate_image_nearest(target_labels, angle) 
        [r,c] = np.shape(target_rotate)
        target_prob = np.zeros((self.num_labels,r,c))
        for n in range(0,self.num_labels):
            target_prob[n,target_rotate==n] = 1 
        return target_prob

    def augment(self,imgs, targets):
        """
        Augmentation of the data which includes: x flip, yflip and rotation, 
        based on the probability to use each method
        """
        values = np.arange(0, 2)
        angle_values = np.arange(-5,6)
        
        is_xflip = np.random.choice(values, p=[1-self.augment_prob_xflip, self.augment_prob_xflip])
        is_yflip = np.random.choice(values, p=[1-self.augment_prob_yflip, self.augment_prob_yflip])
        is_rotate = np.random.choice(values, p=[1-self.augment_prob_rotate, self.augment_prob_rotate])
        
        if is_xflip==1:
           imgs = [np.flip(img,2) for img in imgs] 
           targets = [np.flip(target,2) for target in targets] 
         
        if is_yflip==1:
            imgs = [imgs[n] for n in range(len(imgs)-1,-1,-1)] 
            targets = [targets[n] for n in range(len(targets)-1,-1,-1)] 
        
        if is_rotate==1:
            angle = random.choice(angle_values)
            imgs = [imutils.rotate(img.transpose(1, 2, 0), angle).transpose(2, 0, 1) for img in imgs]
            for n,target in enumerate(targets):
                targets[n] = self.rotate_target(target,angle)
    
        return imgs,targets
    
    def __len__(self):
        self.num_sequences = len(self.sequences_images)
        return self.num_sequences
        
    def __getitem__(self, index):
        """
        returns imgs, targets - a list of sequence images and a list of 
        sequence targets respectively, and seq_name - the name of sequence 
        """
        starting_frame = 1
        seq_name = self.seq_names[index]
        sequence_images = self.sequences_images[index]
        sequence_annotation = self.sequences_annotations[index]

        imgs = []
        targets = []
        for i, (img_path, seg_path) in enumerate(zip(sequence_images, sequence_annotation)):
            img = np.load(img_path).astype(np.float32)
            img_3d = np.broadcast_to(img[:, :, np.newaxis], (512, 500, 3)).transpose(2, 0, 1)
            imgs.append(img_3d)
            target = np.load(seg_path)
            [r,c] = np.shape(target)
            target_prob = np.zeros((self.num_labels,r,c))
            for n in range(0,self.num_labels):
                target_prob[n,target==n] = 1
            targets.append(target_prob)
        
        if self.split == 'train':
           imgs, targets = self.augment(imgs, targets) 
        
        return imgs, targets, seq_name, starting_frame

# a = imgs[0].transpose(1, 2, 0)
# a[a<0] = 0
# a[a>1] = 1
# plt.figure()
# plt.imshow(a)
# plt.show()
# plt.savefig('a.png') 