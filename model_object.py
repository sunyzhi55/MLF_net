from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net.TripleNetwork import *
from Net.MultiLayerFusion import *
from Net.api import *
from loss_function import joint_loss

models = {
'ITCFN':{
        'Name': 'Triple_model_CoAttention_Fusion',
        'Model': Triple_model_CoAttention_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'ITFN':{
        'Name': 'Triple_model_Fusion',
        'Model': Triple_model_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'TFN':{
        'Name': 'Triple_model_Fusion_Incomplete',
        'Model': Triple_model_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'HFBSurv': {
        'Name': 'HFBSurv',
        'Data': './data/summery_new.txt',
        'Batch': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'Dataset_mode': 'fusion',
        'Model': HFBSurv,
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Run': run_main
    },
'TCFN':{
        'Name': 'Triple_model_CoAttention_Fusion_Incomplete',
        'Model': Triple_model_CoAttention_Fusion,
        'Loss': joint_loss,
        'Optimizer': Adam,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },
'MLF':{
        'Name': 'MultiLayerFusionModel',
        'Model': MultiLayerFusionModel,
        'Loss': joint_loss,
        'Optimizer': AdamW,
        'batch_size': 8,
        'Lr': 0.0001,
        'Epoch': 200,
        'w1': 0.2,
        'w2': 0.01,
        'Run': run_main_1
    },

}
