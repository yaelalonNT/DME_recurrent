import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.utils import batch_to_var, make_dir, load_checkpoint, check_parallel, test_prev_mask
from modules.model import RSISMask, FeatureExtractor
from dataloader.dataset_utils import sequence_palette
from PIL import Image
from dataloader.dataset_Hoct import HoctDataset
import torch
import numpy as np
import torch.utils.data as data
from measures.f_boundary import db_eval_boundary as eval_F
from measures.jaccard import db_eval_iou as jaccard_simple
import pickle

class Evaluate():

    def __init__(self,args):

        self.split = args.eval_split
        self.dataset = args.dataset        
        self.dataset = HoctDataset(args,split=self.split)
        self.loader = data.DataLoader(self.dataset, batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         drop_last=False)
        
        self.args = args
        print(args.model_name)
        encoder_dict, decoder_dict, _, _, load_args = load_checkpoint(args.model_name,args.use_gpu)
        load_args.use_gpu = args.use_gpu
        self.encoder = FeatureExtractor(load_args)
        self.decoder = RSISMask(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder,device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder,device_ids=range(args.ngpus))

        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        self.encoder.load_state_dict(encoder_dict)
        
        to_be_deleted_dec = []
        for k in decoder_dict.keys():
            if 'fc_stop' in k:
                to_be_deleted_dec.append(k)
        for k in to_be_deleted_dec:
            del decoder_dict[k]
        self.decoder.load_state_dict(decoder_dict)

        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()
        if load_args.length_clip == 1:
            self.video_mode = False
            print('video mode not activated')
        else:
            self.video_mode = True
            print('video mode activated')
    
    def get_colors(self):
        colors = {0:[0,0,0.6],1:[0.42,0.65,0.016],2:[0.22,0.6,0.83],3:[1,1,0],4:[0,1,0]}# Vitreous,Srf,Irf,Retina,Erm
        return colors

    def save_results(self,x,colors,num_instances,outs,base_dir,starting_frame,ii,GS):
        GS = np.argmax(np.squeeze(GS),0)
        frame_img = x.data.cpu().numpy()[0,:,:,:].squeeze()
        frame_img = np.transpose(frame_img, (1,2,0))
        frame_img[frame_img<0] = 0
        frame_img[frame_img>1] = 1
        label_mask = np.argmax(outs[0,:,:,:].numpy(),axis=0)
        label_map_net = frame_img.copy()
        label_map_GS = frame_img.copy()
        frame_with_overlay = frame_img.copy()
        
        for n in range(0,num_instances):
            label_map_net[label_mask==n,:] = colors[n]
            if n in [1,2,4]: #IRF,SRF,ERM
                frame_with_overlay[label_mask==n,:] = colors[n]

        for n in range(0,num_instances):
            label_map_GS[GS==n,:] = colors[n]    
        
        plt.figure()
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4)    
        fig.set_figheight(20)
        fig.set_figwidth(30)
        
        ax1.imshow(frame_img)
        ax1.set_title('Original image')
        ax1.axis('off')

        ax2.imshow(frame_with_overlay)
        ax2.set_title('Original image with net fluid overlay')
        ax2.axis('off')

        ax3.imshow(label_map_net)
        ax3.set_title('Net label map')
        ax3.axis('off')

        ax4.imshow(label_map_GS)
        ax4.set_title('GS label map')
        ax4.axis('off')
                
        figname = base_dir + 'frame_%02d.png' %(starting_frame[0]+ii)
        plt.savefig(figname,bbox_inches='tight')
        plt.close()
        
    def run_eval(self):
        num_instances = args.maxseqlen
        colors = self.get_colors()
        jaccard_vec, F_vec= [],[]
        
        dev_dir = os.path.join('../../Model', args.model_name, 'validation_results')
        make_dir(dev_dir)
        
        for batch_idx, (inputs, targets,seq_name,starting_frame) in enumerate(self.loader):
            prev_hidden_temporal_list = None
            max_ii = min(len(inputs),args.length_clip)

            base_dir = dev_dir + '/' + seq_name[0] + '/'
            make_dir(base_dir)
                
            for ii in range(max_ii):
                x, y_mask = batch_to_var(args, inputs[ii], targets[ii])
                outs, hidden_temporal_list = test_prev_mask(args, self.encoder, self.decoder, x, prev_hidden_temporal_list)
                outs = outs.reshape(np.shape(y_mask))
                jaccard_vec.append(jaccard_simple(y_mask.cpu().numpy(),outs.cpu().numpy()))
                #F_vec.append(eval_F(y_mask.cpu().numpy(),outs.cpu().numpy()))
                x_tmp = x.data.cpu().numpy()
                height = x_tmp.shape[-2]
                width = x_tmp.shape[-1]
                for t in range(num_instances):
                    mask_pred = (torch.squeeze(outs[0,t,:])).cpu().numpy()
                    mask_pred = np.reshape(mask_pred, (height, width))
                    
                    GS = (torch.squeeze(y_mask[0,t,:])).cpu().numpy()
                    GS = np.reshape(GS, (height, width))
                                                
                    indxs_instance = np.where(mask_pred > 0.5)
                    mask2assess = np.zeros((height,width))
                    maskboll = np.zeros((height,width))
                    mask2assess[indxs_instance] = 255
                    maskboll[indxs_instance] = 1
                    
                    jaccard_vec.append(jaccard_simple(GS,maskboll))
                    F_vec.append(eval_F(GS,maskboll))
                    
                    
                print(seq_name[0] + '/' + '%05d' % (starting_frame[0] + ii))

                self.save_results(x,colors,num_instances,outs,base_dir,starting_frame,ii,y_mask)
                if self.video_mode:
                    prev_hidden_temporal_list = hidden_temporal_list

                del outs, hidden_temporal_list, x

            print('Mean Jaccard index(IOU) = %2f' % (sum(jaccard_vec)/len(jaccard_vec)))
        
        Mean_jaccared = sum(jaccard_vec)/len(jaccard_vec)
        Mean_F = sum(F_vec)/len(F_vec)
        return Mean_jaccared, Mean_F                   
                                  

if __name__ == "__main__":
    data_folder = '2023_05_17_17'
    model_folder = '08_06_23-13'
    
    model_name = os.path.join(data_folder,model_folder)
    args = pickle.load(open(os.path.join('../../Model',model_name,'args.pkl'),'rb'))
    args.use_gpu = torch.cuda.is_available()
    args.model_name = model_name
    args.hoct_dir = os.path.join('../../Model',data_folder)
    args.dataset = 'Hoct' 
    args.eval_split = 'val' 
    args.batch_size = 1 
    args.length_clip = 100 # Entire volume scan
    args.num_workers = 0
    args.maxseqlen = 5 # As the number of labels 

    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    if not args.log_term:
        print ("Eval logs will be saved to:", os.path.join('../../Model',args.model_name, 'eval.log'))
        sys.stdout = open(os.path.join('../../Model',args.model_name, 'eval.log'), 'w')

    E = Evaluate(args)
    Mean_jaccared, Mean_F  = E.run_eval()

    print('Mean jaccared index(IOU)= %.2f, Mean F= %.2f' % (Mean_jaccared,Mean_F))