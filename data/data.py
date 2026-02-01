import torch
import torch.nn as nn



dataset_attributes = {
    'synthetic': {
        'subdir': 'synthetic', 
        'purpose': 'classification',
        'loss': nn.BCEWithLogitsLoss(reduction='mean')
    },

    'adult': { # not included in the paper
        'subdir': 'adult',
        'purpose': 'classification',
        'loss': nn.BCEWithLogitsLoss(reduction='mean')
    },

    'compas': { # not included in the paper
        'subdir': 'compas',
        'purpose': 'classification',
        'loss': nn.BCEWithLogitsLoss(reduction='mean')
    },

    'insurance': { # not included in the paper
        'subdir': 'insurance',
        'purpose': 'regression',
        'loss': nn.MSELoss(reduction='mean')
    },
}