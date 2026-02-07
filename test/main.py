import os
import argparse
import yaml
from solver import Solver


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', required=True)
    parser.add_argument('--input', type=int, default=3, help='annotations of inputs')    # specify annotations
    parser.add_argument('--model', type=str, default='FPN', help='model name')
    # model: [FPN, RED_CNN, SUNet, CTformer, SwinIR]
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--devices', type=int, default=[0,1], nargs='+', help='use devices for data parallel')
    parser.add_argument('--config_path', type=str, default='config', help='config path')
    parser.add_argument('-c', '--config_file', type=str, help='config filename')

    # args from argparse
    opts = parser.parse_args()
    if opts.config_file is None:
        opts.config_file = f'{opts.model}.yaml'
    opts_dict = vars(opts)

    # args from yaml
    yaml_file = os.path.join(opts.config_path, opts.config_file)
    with open(yaml_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)


    args.update(opts_dict)
    cfg = dict_to_namespace(args)

    cfg.annotation = os.path.join(cfg.dataset_path, cfg.dataset_name, 'annotation')
    cfg.log_path = os.path.join(cfg.log_path, cfg.dataset_name, cfg.model)
    # cfg.log_path = f'{cfg.log_path}/{cfg.model}'

    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    if cfg.mode == 'test':
        assert cfg.model == cfg.checkpoint_file[:len(cfg.model)], 'Unmached checkpoint'

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


def main(cfg):
    solver = Solver(cfg)
    if cfg.mode == 'train':
        solver.train()
    elif cfg.mode == 'test':
        save_path = os.path.join(cfg.output_path, cfg.dataset_name, cfg.checkpoint_file[:-3])
        os.makedirs(save_path, exist_ok=True)
        print(f'Output path: {save_path}\n')
        solver.test(save_path)


if __name__ == '__main__':
    cfg = parse()
    print(cfg)

    main(cfg)