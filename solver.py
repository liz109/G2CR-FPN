import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
from tqdm import tqdm
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
import torch.nn.functional as F

from src.model import model_utils
from src.data import data_utils, measure

from src.data.load_data import AAPMDataset

pass_manners = ['high', 'low']
stage_manners = ['low_level','high_level']
ave_spectrum_list = [False, False]


class Solver(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        # Parameter
        self.cfg = cfg

        # Model
        self.model = model_utils.build_model(cfg)
        print("\n##########", cfg.model, "##########")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)   
        print('Model size (grad): {:.7f}MB'.format(total_params / 1024**2))

        # Option
        self.optimizer = model_utils.build_optimizer(self.cfg, self.model)      
        self.scheduler = model_utils.build_scheduler(self.cfg, self.optimizer)
        self.loss_func1 = model_utils.build_loss(self.cfg)      # FF
        self.loss_func2 = torch.nn.MSELoss(reduction='mean')

        # Device
        self.use_gpu = cfg.use_gpu and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{cfg.gpu}' if self.use_gpu else 'cpu')
        self.model.to(self.device)
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
 
        if self.cfg.ckpt_path:
            print("====> Loaded ckpt:", self.cfg.ckpt_path)
            self.ckpt_dict = torch.load(self.cfg.ckpt_path, weights_only=True)
            self.model.load_state_dict(self.ckpt_dict['model_state_dict'])
            # self.optimizer.load_state_dict(self.ckpt_dict["optimizer_state_dict"])


    def loader_dict_(self):
        annotation_file = os.path.join(self.cfg.annotation, f'annotation_{self.cfg.case}.pkl')
        print(f'\n====> Loaded input annotation file: {annotation_file}')
        data = AAPMDataset(annotation_file, resize=self.cfg.resize, \
                           patch_n=self.cfg.patch_n, patch_size=self.cfg.patch_size)

        train_indices_file = os.path.join(self.cfg.annotation, f'train_{self.cfg.case}.npy')
        train_indices = np.load(train_indices_file)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(data, batch_size=self.cfg.batch_size, sampler=train_sampler, num_workers=self.cfg.num_workers, pin_memory=True)

        test_indices_file = os.path.join(self.cfg.annotation, f'test_{self.cfg.case}.npy')
        test_indices = np.load(test_indices_file)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = DataLoader(data, batch_size=self.cfg.batch_size, sampler=test_sampler, num_workers=self.cfg.num_workers, pin_memory=True)

        loader_dict = {'train': train_loader, 'test': test_loader}
        print("Train/Test loader size: {}, {}\n".format(len(train_loader), len(test_loader)))
        return loader_dict


    def compute_loss(self, pred, target):
        lambda1 = 0.1      # FF
        lambda2 = 1         # MSE
        total_loss = 0.0

        ####### FMB #######
        loss_fmb = 0.0
        loss_weight = [1, 1.3]
        for i in range(2):
            _loss = self.loss_func1(pred, target, \
                pass_manner=pass_manners[i], \
                stage_manner=stage_manners[i], \
                ave_spectrum=ave_spectrum_list[i], \
                )
            weighted_loss = _loss * loss_weight[i]
            loss_fmb += weighted_loss

        ##### MSE ####### 
        loss_mse = self.loss_func2(pred, target)

        total_loss = lambda1 * loss_fmb + lambda2 * loss_mse
        return total_loss, loss_fmb, loss_mse
 

    def train(self, save_path):
        # # Log
        # timestamp = str(int(datetime.now().timestamp()))
        # writer = SummaryWriter(os.path.join(self.cfg.log_path, timestamp))
        writer = SummaryWriter(save_path)

        # Train
        step = 0                # global step
        self.ckpt_dict['loss'] = self.ckpt_dict.get('loss', float('inf'))

        start_time = time.time()
        for epoch in range(1, self.cfg.num_epochs + 1):
            ## Train
            self.model.train()
            # running_loss = 0.0
            
            pbar = tqdm(self.loader_dict['train'], desc=f"Epoch {epoch}/{self.cfg.num_epochs}", dynamic_ncols=True)

            for data in pbar:
                step += 1
                images, targets, _ = data
                images, targets = images.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
                # full images: [16, 1, 256, 256]
                # patched images: [16, 10, 1, 64, 64]

                if self.cfg.patch_size: # patch training => [160, 1, 64, 64]
                    images = images.view(-1, 1, self.cfg.patch_size, self.cfg.patch_size)
                    targets = targets.view(-1, 1, self.cfg.patch_size, self.cfg.patch_size)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss, loss_fmb, loss_mse = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # running_loss += loss.item()
                
                # Log info
                lr = self.scheduler.get_last_lr()[0]
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.3e}',
                    'mse': f'{loss_mse.item():.3e}',
                    'lr': f'{lr:.2e}'
                })
            
                if writer:
                    writer.add_scalar('Train/Loss_total', loss.item(), step)
                    writer.add_scalar('Train/Loss_fmb', loss_fmb.item(), step)
                    writer.add_scalar('Train/Loss_mse', loss_mse.item(), step)

            self.scheduler.step()

            # # ======== Val / epoch ========
            # if epoch % 10 == 0:
            avg_loss, avg_loss_fmb, avg_loss_mse = self.val()
            if writer:
                writer.add_scalar('Val/Loss_total', avg_loss, step)
                writer.add_scalar('Val/Loss_fmb', avg_loss_fmb, step)
                writer.add_scalar('Val/Loss_mse', avg_loss_mse, step)

            if avg_loss < self.ckpt_dict['loss']:   
                self.ckpt_dict['loss'] = avg_loss
                self.ckpt_dict['epoch'] = epoch
                self.ckpt_dict['model_state_dict'] = copy.deepcopy(self.model.state_dict())
                self.ckpt_dict['optimizer_state_dict'] = copy.deepcopy(self.optimizer.state_dict())
                
                ckpt_file = os.path.join(save_path, f"tmp.pt")
                torch.save(self.ckpt_dict, ckpt_file)

            self.scheduler.step()

        
        writer.close()
        total_time = (time.time() - start_time)/3600
        print(f'Finished {self.cfg.num_epochs} training epochs in {total_time:.4f} hours.')

        # Save best model
        ckpt_file = os.path.join(save_path, f"{self.ckpt_dict['epoch']}.pt")
        ckpt_file = os.path.join(save_path, "best.pt")
        print(f'====> Saved checkpoint: {ckpt_file}')
        torch.save(self.ckpt_dict, ckpt_file)
        if os.path.exists(os.path.join(save_path, "tmp.pt")):   # remove tmp ckpt file
            os.remove(os.path.join(save_path, "tmp.pt"))


    def val(self):
        self.model.eval()
        running_loss = 0.
        running_loss_fmb, running_loss_mse = 0., 0.
        m = 0
        with torch.no_grad(): 
            for data in tqdm(self.loader_dict['test'], desc='val: '):
                images, targets, _ = data
                images, targets = images.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
                outputs = self.model(images)

                ###################
                loss, loss_fmb, loss_mse = self.compute_loss(outputs, targets)
                ###################
                running_loss += loss.item()
                running_loss_fmb += loss_fmb.item()
                running_loss_mse += loss_mse.item()

                m += 1
            
        avg_loss = running_loss / m     # per batch..
        avg_loss_fmb = running_loss_fmb / m
        avg_loss_mse = running_loss_mse / m
        return avg_loss, avg_loss_fmb, avg_loss_mse
            
            
    def test(self, save_path):
        """
        save_path: path to save results
        """
        # Dataloader
        loader = self.loader_dict['test']

        # compute PSNR, SSIM, RMSE
        img_names = []
        ori_psnrs, ori_ssims, ori_rmses, ori_essims = [], [], [], []
        pred_psnrs, pred_ssims, pred_rmses, pred_essims = [], [], [], []
        
        # Test
        self.model.eval()
        for images, targets, names in tqdm(loader, desc='test: '):
            images, targets = images.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.float)
            outputs = self.model(images)

            images = images.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            # ## denormalize, truncate
            # images = data_utils.denormalize(images, self.cfg.norm_range_min, self.cfg.norm_range_max)
            # images = data_utils.trunc(images, self.cfg.trunc_min, self.cfg.trunc_max)
            # targets = data_utils.denormalize(targets, self.cfg.norm_range_min, self.cfg.norm_range_max)
            # targets = data_utils.trunc(targets, self.cfg.trunc_min, self.cfg.trunc_max)
            # outputs = data_utils.denormalize(outputs, self.cfg.norm_range_min, self.cfg.norm_range_max)
            # outputs = data_utils.trunc(outputs, self.cfg.trunc_min, self.cfg.trunc_max)

            # images = F.interpolate(images, scale_factor=2, mode='bilinear', align_corners=False)
            # targets = F.interpolate(targets, scale_factor=2, mode='bilinear', align_corners=False)
            # outputs = F.interpolate(outputs, scale_factor=2, mode='bilinear', align_corners=False)

            ## denormalize, truncate
            images = data_utils.denormalize(images, self.cfg.trunc_min, self.cfg.trunc_max)
            targets = data_utils.denormalize(targets, self.cfg.trunc_min, self.cfg.trunc_max)
            outputs = data_utils.denormalize(outputs, self.cfg.trunc_min, self.cfg.trunc_max)


            # criterion
            data_range = self.cfg.trunc_max - self.cfg.trunc_min    # 240-(-160)=400.0
            for i in range(len(names)):
                image, target, output, name = images[i].squeeze(0), targets[i].squeeze(0), outputs[i].squeeze(0), names[i]
                
                original_result, pred_result = measure.compute_measure(image, target, output, data_range)
                img_names.append(name)
                ori_rmses.append(original_result[0])
                ori_psnrs.append(original_result[1])
                ori_ssims.append(original_result[2])
                ori_essims.append(original_result[3])

                pred_rmses.append(pred_result[0])
                pred_psnrs.append(pred_result[1])
                pred_ssims.append(pred_result[2])
                pred_essims.append(pred_result[3])

                # ====================== SAVE results ======================
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


        results = {'img_names':img_names, \
            'ori_rmses':ori_rmses, 'ori_psnrs':ori_psnrs, 'ori_ssims':ori_ssims, 'ori_essims':ori_essims,\
            'pred_rmses':pred_rmses, 'pred_psnrs':pred_psnrs, 'pred_ssims':pred_ssims, 'pred_essims':pred_essims}
        results = pd.DataFrame(results)
        results.to_csv(os.path.join(save_path, 'results.csv'))

        stats_path = os.path.join(save_path, 'stas.txt')
        with open(stats_path, 'w') as f:
            # Build output string once
            ori_summary = '\nOriginal === \nRMSE avg: {:.4f}±{:.4f} \nPSNR avg: {:.4f}±{:.4f} \nSSIM avg: {:.4f}±{:.4f} \nESSIM avg: {:.4f}±{:.4f}'.format(
                np.mean(ori_rmses), np.std(ori_rmses),
                np.mean(ori_psnrs), np.std(ori_psnrs),
                np.mean(ori_ssims), np.std(ori_ssims),
                np.mean(ori_essims), np.std(ori_essims)
            )

            pred_summary = '\nPredictions === \nRMSE avg: {:.4f}±{:.4f} \nPSNR avg: {:.4f}±{:.4f} \nSSIM avg: {:.4f}±{:.4f} \nESSIM avg: {:.4f}±{:.4f}'.format(
                np.mean(pred_rmses), np.std(pred_rmses),
                np.mean(pred_psnrs), np.std(pred_psnrs),
                np.mean(pred_ssims), np.std(pred_ssims),
                np.mean(pred_essims), np.std(pred_essims)
                
            )

            # Print to terminal
            print(ori_summary)
            print(pred_summary)

            # Also save to file
            print(ori_summary, file=f)
            print(pred_summary, file=f)
            