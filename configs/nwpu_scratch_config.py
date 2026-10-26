import os

import torch

def get_nwpu_from_scratch_config():
    config = {}

    # --- 1. Dataset & Dataloader ---
    config['data'] = {
        'dataset_type': 'nwpu',
        'nwpu_data_root': '/home/featurize/data/NWPU-RESISC45',
        'nwpu_test_data_root': None,
        'num_classes': 45,
        'img_size': 224,
        'batch_size': 128,
        'val_batch_size': 128,
        'test_batch_size': 128,
        'num_workers': 4,
        'val_split_ratio': 0.3,
        'split_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'split', 'NWPU'),
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
    }

    # --- 2. Model Configuration (VMamba / VSSM) ---
    config['model'] = {
        'name': 'VSSM',
        'params': {
            # input / output
            'img_size': config['data']['img_size'],
            'in_chans': 3,
            'num_classes': config['data']['num_classes'],
            # architecture (基于VMamba-T)
            'depths': [2, 2, 9, 2],
            'dims': 96,
            'drop_path_rate': 0.2,
            'patch_size': 4,
            'patch_norm': True,
            'norm_layer': 'ln',
            'downsample_version': 'v1',
            'patchembed_version': 'v1',
            'posembed': False,
            # SSM
            'ssm_d_state': 16,
            'ssm_ratio': 2.0,
            'ssm_dt_rank': 'auto',
            'ssm_act_layer': 'silu',
            'ssm_conv': 3,
            'ssm_conv_bias': True,
            'ssm_drop_rate': 0.0,
            'ssm_init': 'v0',
            # 阶段扫描设置：per-stage scan: [stage1, stage2, stage3, stage4]
            # 不带z门控： v05_noz/v0_noz=四向交叉扫描  v053d_noz=之字型扫描  v054d_noz=螺旋扫描
            # 带z门控： v05/v0=四向交叉扫描  v053d=之字型扫描  v054=螺旋扫描
            'forward_type': ['v05', 'v05', 'v053d', 'v054d'],
            # MLP
            'mlp_ratio': 0.0,
            'mlp_act_layer': 'gelu',
            'mlp_drop_rate': 0.0,
            'gmlp': False,
            # SAD-CI: spatially-adaptive channel interaction
            'sad_ci': True,
            'sad_ci_mlp_ratio': 2.0,
            'sad_ci_mlp_type': 'gmlp',
            'use_checkpoint': False,
        }
    }

    # --- 3. Training Configuration ---
    config['training'] = {
        'epochs': 500,
        'optimizer': 'AdamW',
        'optimizer_params': {'lr': 5e-4, 'weight_decay': 0.05},
        'scheduler': 'CosineAnnealingLR',
        'scheduler_params': {'T_max': 500, 'eta_min': 1e-6, 'warmup_epochs': 20},
        'criterion': 'CrossEntropyLoss',
        'clip_grad_norm': 1.0,
        'accumulation_steps': 16,  # 梯度累积步数；有效batch = batch_size * accumulation_steps
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'float32',
        'seed': 42,
        'eval_freq': 5,
        'print_freq': 50,
        'checkpoint_dir': './checkpoints',
        'experiment_name': 'nwpu_experiment',
        'max_recent_checkpoints_to_keep': 5,
    }

    return config