# -*- coding: utf-8 -*-

def setup(opt):
    if opt.dataset_name.lower() == 'cmumosei':
        # 从msa_datasets.mosei_reader导入CMUMOSEIReader
        from msa_datasets.mosei_reader import CMUMOSEIReader as MMDataReader
    elif opt.dataset_name.lower() == 'cmumosi':
        # 从msa_datasets.mosi_reader导入CMUMOSIReader
        from msa_datasets.mosi_reader import CMUMOSIReader as MMDataReader
    elif opt.dataset_name.lower() == 'iemocap':
        # 从msa_datasets.iemocap_reader导入IEMOCAPReader
        from msa_datasets.iemocap_reader import IEMOCAPReader as MMDataReader
    else:
        # 默认使用CMUMOSIReader（需确保msa_datasets.mosi_reader存在）
        from msa_datasets.mosi_reader import CMUMOSIReader as MMDataReader

    reader = MMDataReader(opt)
    return reader