# -*- coding:utf-8 -*-
__author__ = 'boredbird'
import os
import numpy as np
import woe.feature_process as fp
import woe.GridSearch as gs
import woe.config as config

import sys
sys.path.append("/usr/local/lib/python2.7/site-packages/IPython/extensions/")
%load_ext autoreload
%autoreload 2

if __name__ == '__main__':
    config_path = os.getcwd()+'//examples/config2.csv'
    data_path = os.getcwd()+'//examples/UCI_Credit_Card.csv'
    feature_detail_path = os.getcwd()+'//examples/features_detail.csv'
    rst_pkl_path = os.getcwd()+'//examples/woe_rule.pkl'
    
    
    cfg = config.config()
    cfg.load_file(config_path,data_path)
    
    # train woe rule
    feature_detail,rst = fp.process_train_woe(infile_path=data_path
                                           ,outfile_path=feature_detail_path
                                           ,rst_path=rst_pkl_path
                                           ,config_path=config_path)
    # proc woe transformation
    woe_train_path = os.getcwd()+'//examples/dataset_train_woed.csv'
    fp.process_woe_trans(data_path,rst_pkl_path,woe_train_path,config_path)
    # here i take the same dataset as test dataset
    woe_test_path = os.getcwd()+'//examples/dataset_test_woed.csv'
    fp.process_woe_trans(data_path,rst_pkl_path,woe_test_path,config_path)

    print '###TRAIN SCORECARD MODEL###'
    params = {}
    params['dataset_path'] = woe_train_path
    params['validation_path'] = woe_test_path
    params['config_path'] = config_path

    params['df_coef_path'] = os.getcwd()+'//df_model_coef_path.csv'
    params['pic_coefpath'] = os.getcwd()+'//model_coefpath.png'
    params['pic_performance'] = os.getcwd()+'//model_performance_path.png'
    params['pic_coefpath_title'] = 'model_coefpath'
    params['pic_performance_title'] = 'model_performance_path'

    params['var_list_specfied'] = []
    params['cs'] = np.logspace(-4, -1,40)
    for key,value in params.items():
        print key,': ',value
    gs.grid_search_lr_c_main(params)
