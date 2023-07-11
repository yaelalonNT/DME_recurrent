import matplotlib
matplotlib.use('Agg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from args import get_parser
import torch
import random
import datetime
from utils.train_utils import trainIters


if __name__ == "__main__":    
    parser = get_parser()
    args = parser.parse_args()
    args.use_gpu = torch.cuda.is_available()
    args.log_term = False
    args.hoct_dir = r'\\nv-nas01\Data\DME_recurrent\Model\2023_05_17_17'
    args.dataset = 'Hoct'
    args.num_workers = 0
    args.max_epoch = 20
    args.length_clip = 5
    args.batch_size = 2
    args.print_every = 100
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
