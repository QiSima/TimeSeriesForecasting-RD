import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import shutil
from tqdm import trange
import statistics
import numpy as np
import torch
from task.TaskLoader import Opt
from task.dataset import de_scale
from task.util import os_makedirs, os_rmdirs, set_logger
import importlib
import math
from tqdm.std import tqdm
from task.util import plot_xfit

class Task(Opt):
    def __init__(self, args):
        if args.attn:
            self.exp_module_path = importlib.import_module('data.{}.{}Infor_attn.H{}'.format(
            args.datafolder.replace('/', '.'), args.dataset,args.H)) 
        else:
            self.exp_module_path = importlib.import_module('data.{}.{}.H{}'.format(
            args.datafolder.replace('/', '.'), args.dataset,args.H))  
            
        self.save_model = args.save_model
        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        if 'Informer' in args.model:
            self.data_opts.info.normal = True
        self.data_subconfig()
        

    def data_config(self, args):
        self.data_name = args.dataset
        if args.attn:
            data_opts = getattr(self.exp_module_path, args.dataset + 'Infor_data')
        else:
            data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        self.data_opts = data_opts(args)

    def data_subconfig(self,):
        self.data_opts.arch = self.model_opts.arch
        self.data_opts.sub_config()

    def model_config(self, args):
        self.model_name = args.model
        if hasattr(self.exp_module_path, self.model_name):
            model_opts = getattr(self.exp_module_path,
                             args.model)
        else:
            try:
                share_module_path = importlib.import_module('data.base')
                model_opts = getattr(share_module_path, self.model_name + '_default')
            except:
                raise ValueError('Non-supported model {} in the data.base module, please check the module or the model name'.format(self.model_name))
                             
        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        self.model_opts.hyper.H = args.H

        if self.model_opts.arch == 'cnn':
            if not self.model_name == 'clstm':
                self.model_opts.hyper.kernel_size = math.ceil(
                    self.model_opts.hyper.steps / 4)

    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        return model

    def exp_config(self, args):
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.device = torch.device('cuda:{}'.format(args.gid))
        else:
            self.device = torch.device('cpu')

        if 'statistic' in vars(self.model_opts):
            self.device = torch.device('cpu')

        self.exp_dir = 'trial' if args.test == False else 'test'
        
        # setting save path
        if args.my_dir:
            self.exp_dir = args.my_dir

        if args.mo is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.mo)

        self.exp_dir = os.path.join(
            self.exp_dir, 'normal') if self.data_opts.info.normal else os.path.join(self.exp_dir, 'minmax')

        assert args.diff == False  

        self.exp_dir = os.path.join(self.exp_dir, args.dataset)

        task_name = os.path.join('{}.refit'.format(args.model), 'h{}'.format(
            args.H)) if 'refit' in self.model_opts.hyper.dict and self.model_opts.hyper.refit else os.path.join('{}'.format(args.model), 'h{}'.format(args.H))

        self.task_dir = os.path.join(self.exp_dir, task_name)

        if args.test and args.logger_level != 20:
            self.logger_level = 50  
        else:
            self.logger_level = 20  

        self.rep_times = args.rep_times

        if args.clean:
            os_rmdirs(self.task_dir)
        os_makedirs(self.task_dir)

        self.model_opts.hyper.device = self.device

    def logger_config(self, dir, stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage, cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger

    def conduct(self,):
        # init and mkdir taskdir
        # generate the subPack dataset
        for sub_count, series_Pack in enumerate(tqdm(self.data_opts.seriesPack)):
            assert sub_count == series_Pack.index
            self.series_dir = os.path.join(
                self.task_dir, 'series{}'.format(sub_count))
            self.measure_dir = os.path.join(self.series_dir, 'eval_results')
            os_makedirs(self.measure_dir)

            for i in trange(self.rep_times):

                result_file = os.path.join(
                    self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
                

                if os.path.exists(result_file):
                    continue
                if i > 0 and 'statistic' in self.model_opts.dict:
                    assert self.model_opts.statistic
                    result0 = str(os.path.join(
                        self.measure_dir, 'results_{}.series_{}.npy'.format(0, sub_count)))
                    shutil.copy(result0, result_file)
                    continue

                cLogger = self.logger_config(
                    self.series_dir, 'train', i, sub_count)
                cLogger.critical('*'*80)
                cLogger.critical('Dataset: {}\t Model:{} \t InputL:{}\t H: {}\t Trail: {} \t'.format(
                    self.data_name, self.model_name, self.data_opts.info.steps,self.model_opts.hyper.H, i))

                self.conduct_iter(i, series_Pack, result_file, cLogger)

    def conduct_iter(self, i, subPack, result_file, clogger):
        try:

            clogger.critical(
                'For {}th-batch-trainingLoading, loading the sub-datasets {}'.format(i, subPack.index))
            clogger.critical('-'*80)

            self.model_opts.hyper.H = subPack.H
            self.model_opts.hyper.series_dir = self.series_dir
            self.model_opts.hyper.sid = subPack.index
            self.model_opts.hyper.cid = i
            self.model_opts.hyper.data_name = self.data_name
            self.model_opts.hyper.PreTrained_dir = self.exp_dir
            

            model = self.model_import()
            model = model(self.model_opts.hyper, clogger)

            clogger.critical('Loading complete.')
            clogger.critical(f'Model: \n{str(model)}')
            
            fit_info = model.xfit(subPack.train_loader, subPack.valid_loader)
            
            if self.save_model:
                if self.model_opts.arch == 'atten_base' or self.model_opts.arch == 'seq2seq':
                    _path = '{}/models'.format(self.series_dir)
                    if not os.path.exists(_path):
                        os.makedirs(_path)
                    model.logger = None
                    if self.model_name == 'seq2seqRL':
                        torch.save({'policy': model.best_policy,'environment':model.best_environment},'{}/{}_cv{}.pkl'.format(_path,self.model_name,self.model_opts.hyper.cid))
                    else:
                        torch.save(model.state_dict(),'{}/{}_cv{}.pkl'.format(_path,self.model_name,self.model_opts.hyper.cid))
                        
            
            if isinstance(fit_info,Opt): 
                self.plot_fitInfo(fit_info, subId=subPack.index,
                                cvId=i, flogger=clogger)
                
            _, tgt, pred = model.loader_pred(subPack.test_loader)
            
            if isinstance(pred,list): 
                pred = pred[0]
                
            _tgt, _pred = de_scale(
            subPack, tgt), de_scale(subPack, pred)

            clogger.critical('-'*50)
            
            np.save(result_file,
                    (_tgt, _pred))
            
            
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def evaluation(self, metrics=['rmse'], force_update=True):
        try:
            self.metrics = metrics
            eval_list = []
            eLogger = set_logger(os.path.join(self.task_dir, 'eval.log'), '{}.H{}.{}'.format(
                self.data_name, self.data_opts.info.H, self.model_name.upper()), self.logger_level)
            
            
            for sub_count in range(self.data_opts.info.num_series):

                ser_eval = []
                self.series_dir = os.path.join(
                    self.task_dir, 'series{}'.format(sub_count))
                self.measure_dir = os.path.join(
                    self.series_dir, 'eval_results')
                os_makedirs(self.measure_dir)

                for i in range(self.rep_times):
                    metric_file = os.path.join(
                        self.measure_dir, 'metrics_{}.series_{}.npy'.format(i, sub_count))

                    if os.path.exists(metric_file) and force_update is False:
                        eval_results = np.load(metric_file)
                    else:
                        eval_results = self.eval_iter(
                            i, sub_count)

                    eLogger.critical('*'*80)
                    eLogger.critical('Dataset: {}\t Model: {}\t H: {}\tSeries-id: {}\t Trail-id: {}'.format(
                        self.data_name, self.model_name, self.data_opts.info.H, sub_count, i))
                    for _i, eval_name in enumerate(self.metrics):
                        eLogger.critical(
                            'Testing\t{}:\t{:.4g}'.format(eval_name, eval_results[0, _i]))
                    ser_eval.append(eval_results)
                    np.save(metric_file, eval_results)
                eval_list.append(ser_eval)
                eLogger.critical('-'*80)

            self.eval_info = Opt()
            self.eval_info.series = []

            for sub_count, ser_eval in enumerate(eval_list):
                eLogger.critical('='*80)
                eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-id: {} \t Trail-Nums: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.H, sub_count, self.rep_times))

                series_eval_dict = self.eval_list2dict(ser_eval)
                self.eval_info.series.append(series_eval_dict)
                for metric_name in self.metrics:
                    eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                        metric_name, series_eval_dict[metric_name]['mean'], series_eval_dict[metric_name]['std']))
            
            eLogger.critical('@'*80)
            eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-Nums: {}\t Trail-Nums: {}'.format(
                self.data_name, self.model_name, self.data_opts.info.H, self.data_opts.info.num_series, self.rep_times))

            all_eval_list = [item for series in eval_list for item in series]
            eval_return = self.eval_list2dict(all_eval_list)
            for metric_name in self.metrics:
                eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                    metric_name, eval_return[metric_name]['mean'], eval_return[metric_name]['std']))
            
            self.eval_info.all = eval_return
            
            
            return self.eval_info
        except:
            eLogger.exception(
                '{}\nGot an error on evaluation.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_iter(self, i, sub_count):
        result_file = os.path.join(
            self.measure_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
        _test_target, _pred = np.load(result_file)
        eval_results_len = len(self.metrics)
        eval_results = np.zeros((1, eval_results_len))
        for i, eval_name in enumerate(self.metrics):
            measure_path = importlib.import_module('task.metric')
            eval = getattr(measure_path, eval_name)
            if eval_name == 'mase':
                eval_result = eval(_test_target, _pred, self.data_opts.seriesPack[sub_count].avgNaiveError)
            else:
                eval_result = eval(_test_target, _pred)
            eval_results[0, i] = eval_result
        return eval_results

    def get_naivePred(self, subcount):
        subPack = self.data_opts.seriesPack[subcount]
        testloader =subPack.test_loader
        
        tx = []
        for batch_x, _ in testloader:
            tx.append(batch_x)
        tx = torch.cat(tx, dim=0).detach().cpu().numpy()
        if len(tx.shape) == 3:
            tx = tx[:, 0, :]
        else:
            tx = np.expand_dims(tx[:,0],1)
        
        _tx = de_scale(subPack, tx, tag='input')
        
        _pred = _tx[:, -1]
        return _pred        
    
    def plot_fitInfo(self, fit_info, subId, cvId, flogger):
        if 'loss_list' in fit_info.dict and 'vloss_list' in fit_info.dict:
            plot_dir = os.path.join(self.series_dir, 'figures')
            os_makedirs(plot_dir)

            plot_xfit(fit_info,'cv{}.series{}'.format(cvId, subId), plot_dir)
            flogger.critical('Ploting complete. Saving in {}'.format(plot_dir))

    def eval_list2dict(self, _eval_list):
        eval_data = np.concatenate(_eval_list, axis=0)

        eval_return = {}
        for i, metric_name in enumerate(self.metrics):
            i_data = eval_data[:, i].tolist()
            if len(eval_data) > 1:
                mean = statistics.mean(i_data)
                std = statistics.stdev(i_data, mean)
            else:
                mean = i_data[0]
                std = 0
            _min = min(i_data)

            eval_return[metric_name] = {}
            eval_return[metric_name]['mean'] = mean
            eval_return[metric_name]['std'] = std
            eval_return[metric_name]['raw'] = i_data
            eval_return[metric_name]['min'] = _min
            
        return eval_return
