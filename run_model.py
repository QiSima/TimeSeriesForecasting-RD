import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.parser import get_parser
from task.TaskWrapper import Task 
import torch
import gc

if __name__ == "__main__":             
    
    args = get_parser(parsing=False)
    
    args.add_argument('-sid',type= int, default=0, help='experimental series id' )
    args.add_argument('-cid', type= int, nargs='+', default=[0], help='experimental cross validation id')

    args = args.parse_args()
    args.cuda = True
    args.clean = True
    args.datafolder = 'seq_lstm' # seq_lstm/seq_informer/seq_eslstm
    args.my_dir = 'results_lstm' #  results_lstm/results_informer/results_eslstm

    args.attn = False # True for seq_informer
    model_list = ['Seq2_RL']  # Seq2_RL/Informer_RL/ESLSTM_RL
    data_list = ['ili','ili','pm','pm','SML1','SML1','ETTh1','ETTh1','mg','mg']
    H_list = [4,12,30,60,48,96,24,48,17,84]
    
    args.test = False
    args.rep_times = 1
    args.save_model = True

    try:
        for k in range(len(data_list)):
            for i in range(len(model_list)):
                args.model = model_list[i]
                args.dataset = data_list[k]
                args.H = H_list[k]
                
                task = Task(args)
                task.conduct()

                args.metrics = ['rmse','mape']
                task.evaluation(args.metrics)
    except Exception as e:
        print(e)
        torch.cuda.empty_cache() if args.cuda else gc.collect()