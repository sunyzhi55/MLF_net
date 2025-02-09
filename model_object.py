from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net.MultiLayerFusion import *
from Net.api import *
from loss_function import joint_loss

models = {
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
