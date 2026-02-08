import os
import argparse
import yaml
from datetime import datetime

from solver import Solver
# from solver_ge import Solver

"""
# PRE: annotation.py to link data paths

# TRAIN
python main.py --mode train --case 0 --devices 0 1 --config_path ./config/FPN.yaml 

# FINE TUNE with ckpt
python main.py --mode train --case 0 --devices 0 1 \
    --ckpt_path output/aapm_all_npy_3mm/FPN_1769208272/best.pt
    
# TEST
python main.py --mode test --case 0 --devices 0 1 \
    --ckpt_path output/aapm_all_npy_3mm/FPN_1769208272/best.pt

"""

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', required=True)
    parser.add_argument('--case', type=int, default=1, help='annotation case')    # specify annotations
    parser.add_argument('--model', type=str, default='FPN', help='model name')
    # model: [FPN, RED_CNN, SUNet, CTformer, SwinIR]
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--devices', type=int, default=[0,1], nargs='+', help='use devices for data parallel')
    parser.add_argument('--config_path', type=str, default='./config/FPN.yaml', help='config file *.yaml')
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint file *.pt. If test, must provide!!!!')
    

    # args from argparse
    opts = parser.parse_args()
    if opts.mode == 'test':        
        ckpt_dir = os.path.dirname(opts.ckpt_path)
        opts.config_path = os.path.join(ckpt_dir, f'{opts.model}.yaml')
    opts_dict = vars(opts)

    # args from yaml
    print(f'\n==> Loading config file: {opts.config_path}\n')
    yaml_file = opts.config_path
    with open(yaml_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    args.update(opts_dict)
    cfg = dict_to_namespace(args)

    cfg.annotation = os.path.join(cfg.dataset_path, cfg.dataset_name)

    # if cfg.mode == 'test':
    #     filename = os.path.basename(cfg.ckpt_path)
    #     assert cfg.model == filename[:len(cfg.model)], 'Unmached checkpoint'

    # # Print cfg
    # for k, v in vars(cfg).items():
    #     print(f'{k}: {v}')
    return cfg


# Function to convert a nested dictionary to a nested argparse.Namespace
def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def namespace_to_dict(namespace):
    if isinstance(namespace, argparse.Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(namespace).items()}
    elif isinstance(namespace, dict):
        return {k: namespace_to_dict(v) for k, v in namespace.items()}
    else:
        return namespace
    

def main(cfg):
    
    if cfg.mode == 'train':
        # ========== Create output directory ==========
        timestamp = str(int(datetime.now().timestamp()))
        save_path = os.path.join(cfg.output_path, cfg.dataset_name, f'{cfg.model}_{timestamp}')
        os.makedirs(save_path, exist_ok=False)
        print(f'\n==> Output: {save_path}')

        # Save config to YAML
        config_save_path = os.path.join(save_path, f'{cfg.model}.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(namespace_to_dict(cfg), f)
        print(f"====> Saved config: {config_save_path}\n")
        
        solver = Solver(cfg)
        solver.train(save_path)
        
    elif cfg.mode == 'test':

        if not cfg.ckpt_path or not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"\n==> Checkpoint file does not exist: {cfg.ckpt_path}")
        
        save_path = os.path.join(os.path.dirname(cfg.ckpt_path), f'test_{cfg.case}')
        os.makedirs(save_path, exist_ok=False)
        print(f'\n==> Output (in the checkpoint path): {save_path}')
        
        solver = Solver(cfg)
        solver.test(save_path)


if __name__ == '__main__':
    cfg = parse()

    main(cfg)