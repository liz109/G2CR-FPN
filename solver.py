import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import tqdm
import random
import numpy as np
import pandas as pd
import shutil
import copy
from datetime import datetime
import time
import math
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.load_data import AAPMDataset
from src.model import model_utils
from src.data import data_utils, measure


class Solver(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        # Parameter
        self.cfg = cfg

        # Model
        self.model = model_utils.build_model(cfg)
        print("##########", cfg.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)   
        print('Model size (grad): {:.7f}MB'.format(total_params / 1024**2))


        # Device
        self.use_gpu = cfg.use_gpu and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{cfg.gpu}' if self.use_gpu else 'cpu')
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=cfg.devices)
        if torch.cuda.device_count() >= len(cfg.devices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.devices)
        
        # Seed
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = True
        if self.use_gpu:
            torch.cuda.manual_seed_all(cfg.seed)

        # Dataloader
        self.loader_dict = self.loader_dict_()

        # Checkpoint
        self.ckpt_dict = {
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'epoch': 0,
            'loss': float('inf')
        }
    

    def loader_dict_(self):
        annotation_file = os.path.join(self.cfg.annotation, f'annotation_{self.cfg.input}.pkl')
        print('\n')
        print(f'Annotation file: {annotation_file}\n')
        data = AAPMDataset(annotation_file, resize=self.cfg.resize, \
                           patch_n=self.cfg.patch_n, patch_size=self.cfg.patch_size)

        train_indices_file = os.path.join(self.cfg.annotation, f'train_{self.cfg.input}.npy')
        train_indices = np.load(train_indices_file)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(data, batch_size=self.cfg.batch_size, sampler=train_sampler, num_workers=self.cfg.num_workers, pin_memory=True)

        test_indices_file = os.path.join(self.cfg.annotation, f'test_{self.cfg.input}.npy')
        test_indices = np.load(test_indices_file)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(data, batch_size=self.cfg.batch_size, sampler=test_sampler, num_workers=self.cfg.num_workers, pin_memory=True)

        loader_dict = {'train': train_loader, 'test': test_loader}
        print("Train/Test loader size: {}, {}\n".format(len(train_loader), len(test_loader)))
        return loader_dict
    

    def train(self):
        timestamp = str(int(datetime.now().timestamp()))

        # Log
        writer = SummaryWriter(os.path.join(self.cfg.log_path, timestamp))

        # Option
        optimizer = model_utils.build_optimizer(self.cfg, self.model)      
        scheduler = model_utils.build_scheduler(self.cfg, optimizer)
        loss_func = model_utils.build_loss(self.cfg)

        # Train
        start_time = time.time()
        step = 0
        for epoch in range(1, self.cfg.num_epochs + 1):
            ## Train
            self.model.train()
            for data in tqdm.tqdm(self.loader_dict['train'], desc='train: '):
                step += 1
                images, targets, _ = data
                images, targets = images.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
                # full images: [16, 1, 256, 256]
                # patched images: [16, 10, 1, 64, 64]

                if self.cfg.patch_size: # patch training => [160, 1, 64, 64]
                    images = images.view(-1, 1, self.cfg.patch_size, self.cfg.patch_size)
                    targets = targets.view(-1, 1, self.cfg.patch_size, self.cfg.patch_size)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_func(outputs, targets)
                # ####### self-defined new loss 
                # loss1 = loss_func(outputs, targets)
                # loss1 = -math.log(loss1 + 1e-10)  # Adding a small constant to avoid log(0)
                # loss1 = loss1 / max(-math.log(1e-10), -math.log(1.0))   # rescale to [0,1]
                # loss2 = measure.compute_SSIM(outputs, targets, 1)

                # loss = 1-(0.5*loss1 + 0.5*loss2)
                # loss = torch.tensor(loss, requires_grad=True)
                # loss = loss.to(self.device, dtype=torch.float)
                # ########
                loss.backward()
                optimizer.step()

                if step % self.cfg.print_freq == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f'epoch: {epoch}/{self.cfg.num_epochs} | loss: {loss.item():.5e} | lr: {lr:.5e}')
                    # print(f'epoch: {epoch}/{self.cfg.num_epochs} | loss: {loss.item():.5e}')
                writer.add_scalar('Loss/train', loss, step)
                # torch.cuda.empty_cache()

            ## Val / Test
            self.model.eval()
            running_loss = 0.
            m = 0
            with torch.no_grad():
                for data in tqdm.tqdm(self.loader_dict['test'], desc='val: '):
                    images, targets, _ = data
                    images, targets = images.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
                    outputs = self.model(images)
                    loss = loss_func(outputs, targets)      # loss=MSE
                    running_loss += loss.item()
                    m += 1
                
            avg_loss = running_loss / m
            print(f'epoch: {epoch}/{self.cfg.num_epochs} | val_loss: {avg_loss:.3e}')
            writer.add_scalar('Loss/validation', avg_loss, step)

            if avg_loss < self.ckpt_dict['loss'] and epoch > 20:
                self.ckpt_dict['loss'] = avg_loss
                self.ckpt_dict['epoch'] = epoch
                self.ckpt_dict['model_state_dict'] = copy.deepcopy(self.model.state_dict())
                self.ckpt_dict['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())

            print('epoch: {} | best_loss: {:.3e}'.format(self.ckpt_dict['epoch'], self.ckpt_dict['loss']))

            scheduler.step()
        writer.close()
        end_time = time.time()
        total_time = (end_time - start_time)/3600
        print(f'Finished {self.cfg.num_epochs} training epochs in {total_time:.4f} hours.')

        # Save best model
        ckpt_file = "{}_{}_{}.pt".format(self.cfg.model, self.ckpt_dict['epoch'], timestamp)
        ckpt_path = os.path.join(self.cfg.checkpoint_path, self.cfg.dataset_name)
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_file = os.path.join(ckpt_path, ckpt_file)
        print(f'Checkpoint file: {ckpt_file}')
        torch.save(self.ckpt_dict, ckpt_file)


    def test(self, save_path):
        # Dataloader
        loader = self.loader_dict['test']

        # Model
        ckpt_file = os.path.join(self.cfg.checkpoint_path, self.cfg.dataset_name, self.cfg.checkpoint_file)
        print("##########ckpt_file:", ckpt_file)
        self.ckpt_dict = torch.load(ckpt_file)
        self.model.load_state_dict(self.ckpt_dict['model_state_dict'])

        # compute PSNR, SSIM, RMSE
        img_names = []
        ori_psnrs, ori_ssims, ori_rmses = [], [], []
        pred_psnrs, pred_ssims, pred_rmses = [], [], []
        
        # Test
        self.model.eval()
        for images, targets, names in tqdm.tqdm(loader, desc='test: '):
            images, targets = images.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
            outputs = self.model(images)

            images = images.cpu().data
            targets = targets.cpu().data
            outputs = outputs.cpu().data

            ## denormalize, truncate
            images = data_utils.denormalize(images, self.cfg.norm_range_max, self.cfg.norm_range_min)
            images = data_utils.trunc(images, self.cfg.trunc_max, self.cfg.trunc_min)
            targets = data_utils.denormalize(targets, self.cfg.norm_range_max, self.cfg.norm_range_min)
            targets = data_utils.trunc(targets, self.cfg.trunc_max, self.cfg.trunc_min)
            outputs = data_utils.denormalize(outputs, self.cfg.norm_range_max, self.cfg.norm_range_min)
            outputs = data_utils.trunc(outputs, self.cfg.trunc_max, self.cfg.trunc_min)

            # criterion
            data_range = self.cfg.trunc_max - self.cfg.trunc_min    # 400.0
            for i in range(len(names)):
                image, target, output, name = images[i].squeeze(0), targets[i].squeeze(0), outputs[i].squeeze(0), names[i]
                
                original_result, pred_result = measure.compute_measure(image, target, output, data_range)
                img_names.append(name)
                ori_psnrs.append(original_result[0])
                ori_ssims.append(original_result[1])
                ori_rmses.append(original_result[2])
                pred_psnrs.append(pred_result[0])
                pred_ssims.append(pred_result[1])
                pred_rmses.append(pred_result[2])

                if save_path:
                    path = os.path.join(save_path, f'{name}.png')
                    data_utils.save_fig(image, target, output, path, original_result, pred_result, self.cfg.trunc_max, self.cfg.trunc_min)
                    path = os.path.join(save_path, f'{name}_ouput.npy')
                    with open(path, 'wb') as f:
                        np.save(f, output)
                    path = os.path.join(save_path, f'{name}_input.npy')
                    with open(path, 'wb') as f:
                        np.save(f, image)
                    path = os.path.join(save_path, f'{name}_target.npy')
                    with open(path, 'wb') as f:
                        np.save(f, target)


        results = {'img_names':img_names, 'ori_psnrs':ori_psnrs, 'ori_ssims':ori_ssims, 'ori_rmses':ori_rmses,\
                   'pred_psnrs':pred_psnrs, 'pred_ssims':pred_ssims, 'pred_rmses':pred_rmses}
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(save_path, 'results.csv'))

        print('\n')
        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(np.mean(ori_psnrs), 
                                                                                        np.mean(ori_ssims), 
                                                                                        np.mean(ori_rmses)))
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f}±{:.4f} \nSSIM avg: {:.4f}±{:.4f} \nRMSE avg: {:.4f}±{:.4f}'.format(\
            np.mean(pred_psnrs), np.std(pred_psnrs),
            np.mean(pred_ssims), np.std(pred_ssims),
            np.mean(pred_rmses), np.std(pred_rmses)))
