import matplotlib
matplotlib.use('Agg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
from args import get_parser
from modules.model import RSISMask, FeatureExtractor
from utils.utils import get_optimizer, batch_to_var, make_dir, check_parallel
from utils.utils import save_checkpoint_prev_mask, load_checkpoint, get_base_params,get_skip_params
from dataloader.dataset_Hoct import HoctDataset
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from utils.objectives import softIoULoss
import time
import pickle
import random
import datetime
from datetime import timedelta

def save_loss_plot(save_path, train_loss, dev_loss):
  plt.figure(figsize = (14,6))
  plt.plot(np.arange(len(dev_loss)), dev_loss, label='Val IOU')
  plt.plot(np.arange(len(train_loss)), train_loss, label='Train IOU')
  plt.ylim(0, 1)
  plt.xlim(0, len(dev_loss)-1)
  plt.xlabel('# Epoch')
  plt.ylabel('IOU')
  plt.legend()
  plt.grid()
  plt.title('IOU')
  plt.show()
  plt.savefig(os.path.join(save_path,'IOU_convergence.png'))

def collate_fn(batch):
    return tuple(zip(*batch))

def init_dataloaders(args):
    loaders = {}

    # Load train and dev data loaders
    for split in ['train', 'val']:
        batch_size = args.batch_size
        dataset = HoctDataset(args,split)
        loaders[split] = data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         drop_last=True)
    return loaders

def runIter(args, encoder, decoder, x, y_mask,
            crits, optims, mode='train', loss = None, prev_hidden_temporal_list=None, last_frame=False):
    """
    Runs forward a batch
    """
    mask_siou = crits
    enc_opt, dec_opt = optims
    T = args.maxseqlen
    hidden_spatial = None
    out_masks = []
    if mode == 'train':
        encoder.train(True)
        decoder.train(True)
    else:
        encoder.train(False)
        decoder.train(False)
    feats = encoder(x)

    hidden_temporal_list = []

    # loop over number of labels and get predictions
    for t in range(0, T):
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
        else:
            hidden_temporal = None
            
        out_mask, hidden = decoder(feats, hidden_spatial, hidden_temporal)
            
        hidden_tmp = []
        for ss in range(len(hidden)):
            hidden_tmp.append(hidden[ss][0])
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # get predictions in list to concat later
        out_masks.append(out_mask)

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_mask.size(0),len(out_masks), -1)
    out_masks = out_masks.reshape(np.shape(y_mask))
    if not args.use_gpu:
        out_masks = out_masks.contiguous()
    
    #loss is masked with sw_mask
    loss_mask_iou = mask_siou(y_mask.view(-1,y_mask.size()[-1]),out_masks.view(-1,out_masks.size()[-1]))
    loss_mask_iou = torch.mean(loss_mask_iou)

    if loss is None:
        loss = loss_mask_iou
    else:
        loss += loss_mask_iou
                   
    losses = [loss.data.item(), loss_mask_iou.data.item()]
    outs = torch.sigmoid(out_masks)
    outs = outs.data

    return loss, losses, outs, hidden_temporal_list, out_masks
    

def trainIters(args):

    epoch_resume = 0
    model_dir = os.path.join(args.models_path, args.model_name)

    if args.resume: # will resume training the model with name args.model_name
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name,args.use_gpu)
        epoch_resume = load_args.epoch_resume
        encoder = FeatureExtractor(load_args)
        decoder = RSISMask(load_args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)
        args = load_args
    else:
        encoder = FeatureExtractor(args)
        decoder = RSISMask(args)

    make_dir(model_dir)
    pickle.dump(args, open(os.path.join(model_dir,'args.pkl'),'wb'))
    encoder_params = get_base_params(args,encoder)
    skip_params = get_skip_params(encoder)
    decoder_params = list(decoder.parameters()) + list(skip_params)
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params, args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params, args.weight_decay_cnn)
    
    if args.resume:
        enc_opt.load_state_dict(enc_opt_dict)
        dec_opt.load_state_dict(dec_opt_dict)
        from collections import defaultdict
        dec_opt.state = defaultdict(dict, dec_opt.state)

    # objective function for mask
    mask_siou = softIoULoss()

    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()
        mask_siou.cuda()

    crits = mask_siou
    optims = [enc_opt, dec_opt]
    if args.use_gpu:
        torch.cuda.synchronize()
    start = time.time()

    # vars for early stopping
    best_val_loss = args.best_val_loss
    acc_patience = 0
    mt_val = -1
    loaders = init_dataloaders(args)
    num_batches = {'train': 0, 'val': 0}
    
    epochs_loss = {'train':[],'val':[]}
 
    for e in range(args.max_epoch):
        print ("Epoch", e + epoch_resume)
        # store losses in lists to display average since beginning
        epoch_losses = {'train': {'total': [], 'iou': []},
                            'val': {'total': [], 'iou': []}}
        
        # check if it's time to do some changes here
        if e + epoch_resume >= args.finetune_after and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            args.update_encoder = True
            acc_patience = 0
            mt_val = -1

        # we validate after each epoch
        for split in ['train', 'val']:
            Batches = len(loaders[split])                    
            for batch_idx, (inputs, targets, seq_name,starting_frame) in enumerate(loaders[split]):                    
                # send batch to GPU                    
                prev_hidden_temporal_list = None
                loss = None
                last_frame = False
                max_ii = min(len(inputs),args.length_clip)                      
                                    
                for ii in range(max_ii):
                    if ii == max_ii-1:
                        last_frame = True
                    x, y_mask = batch_to_var(args, inputs[ii], targets[ii])
                        
                    loss, losses, outs, hidden_temporal_list,out_masks = runIter(args, encoder, decoder, x, y_mask,
                                                                       crits, optims, split,
                                                                       loss, prev_hidden_temporal_list, last_frame)
                    if last_frame: #Backpropagate in the last frame
                        enc_opt.zero_grad()
                        dec_opt.zero_grad()
                        decoder.zero_grad()
                        encoder.zero_grad()
                    
                        if split == 'train':
                            loss.backward()
                            dec_opt.step()
                            if args.update_encoder:
                                enc_opt.step()
                                                                                    
                    if last_frame:
                        loss = None
                    prev_hidden_temporal_list = hidden_temporal_list

                # store loss values in dictionary separately
                epoch_losses[split]['total'].append(losses[0])
                epoch_losses[split]['iou'].append(losses[1])
                
                # print after some iterations
                if (batch_idx + 1)% args.print_every == 0:      
                    mt = np.mean(epoch_losses[split]['total'])
                    mi = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    te = str(timedelta(seconds=te))
                    print_str = "%s Epoch %d : Batch %d/%d, total IOU loss = %.2f, IOU loss = %.4f, time:%s" % (split,e+1,batch_idx,Batches, mt, mi, te) 
                    print(print_str)
                    
                    text_file = open(model_dir + '/Training_logs.txt', "a")
                    text_file.write('\n' + print_str)
                    text_file.close()
                
                if args.use_gpu:
                    torch.cuda.synchronize()
            num_batches[split] = batch_idx + 1
            if split == 'val' and args.smooth_curves:
                if mt_val == -1:
                    mt = np.mean(epoch_losses[split]['total'])
                else:
                    mt = 0.9*mt_val + 0.1*np.mean(epoch_losses[split]['total'])
                mt_val = mt

            else:
                mt = np.mean(epoch_losses[split]['total'])

            mi = np.mean(epoch_losses[split]['iou'])
            epochs_loss[split].append(1-mi) # save train and val IOU for the epoch
            args.epoch_resume = e + epoch_resume

        save_checkpoint_prev_mask(args, model_dir, e,  encoder, decoder, enc_opt, dec_opt,isbest = 0)    
        save_loss_plot(model_dir, epochs_loss['train'], epochs_loss['val'])
        with open(os.path.join(model_dir,'IOU.pickle'), 'wb') as handle:
            pickle.dump(epochs_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        if mt < (best_val_loss - args.min_delta):
            print ("Saving checkpoint.")
            best_val_loss = mt
            args.best_val_loss = best_val_loss
            # saves model, params, and optimizers
            save_checkpoint_prev_mask(args, model_dir, e,  encoder, decoder, enc_opt, dec_opt, isbest = 1)
            acc_patience = 0
        else:
            acc_patience += 1


        if acc_patience > args.patience and not args.update_encoder and not args.finetune_after == -1:
            print("Starting to update encoder")
            acc_patience = 0
            args.update_encoder = True
            best_val_loss = 1000  # reset because adding a loss term will increase the total value
            mt_val = -1
            encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, _ = load_checkpoint(args.model_name,args.use_gpu)
            encoder.load_state_dict(encoder_dict)
            decoder.load_state_dict(decoder_dict)
            enc_opt.load_state_dict(enc_opt_dict)
            dec_opt.load_state_dict(dec_opt_dict)

        # early stopping after N epochs without improvement
        if acc_patience > args.patience_stop:
            break


if __name__ == "__main__":    
    parser = get_parser()
    args = parser.parse_args()
    args.use_gpu = 0
    args.log_term = False
    args.hoct_dir = r'\\nv-nas01\Data\DME_recurrent\Model\2023_05_17_15'
    args.dataset = 'Hoct'
    #args.dataset = 'davis2017'
    args.num_workers = 0
    args.max_epoch = 20
    args.length_clip = 5
    args.batch_size = 4
    args.print_every = 1
    args.maxseqlen = 5 # As the number of labels 
    
    args.models_path = args.hoct_dir
    if not os.path.isdir(args.models_path):
        os.mkdir(args.models_path)
    now = datetime.datetime.now()
    current_time = now.strftime("%d_%m_%y-%H")
    args.model_name = current_time  
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
        
    trainIters(args)
