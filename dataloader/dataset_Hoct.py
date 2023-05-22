# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:17:04 2023

@author: yael
"""
import pandas as pd
import os
import numpy as np

class HoctDataset():
    def __init__(self,args,split):
        self.data_csv_name = 'data_summary.csv'
        self.length_clip = args.length_clip # Sequence length for training
        self.split = split 
        self.num_labels = args.maxseqlen
        self.data_dir = args.hoct_dir
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
                else: # End of sequence
                    ind = range(len(dir_files)-self.length_clip,len(dir_files))
                    
                seq_img_path = [os.path.join(scan_path_img,str(dir_numbers[f]) + '.npy') for f in ind]
                seq_seg_path = [os.path.join(scan_path_seg,str(dir_numbers[f]) + '.npy') for f in ind]
                seq_name = '%s_%d_%d' %(scan,dir_numbers[ind[0]],dir_numbers[ind[-1]])
                self.sequences_images.append(seq_img_path)
                self.sequences_annotations.append(seq_seg_path)
                self.seq_names.append(seq_name)
        return
    
    
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
            
        return imgs, targets, seq_name, starting_frame
