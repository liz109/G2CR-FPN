import torch
from src.model.fpn import FPN
from src.model.red_cnn import RED_CNN
from src.model.sunet.sunet import SUNet_model
from src.model.ctformer.ctformer import CTformer_model
from src.model.swinir.swinir import SwinIR_model

def build_model(cfg):
    if cfg.model == 'RED_CNN':
        model = RED_CNN()
    elif cfg.model == 'FPN':
        model = FPN(
            C=cfg.FPN.C,
            H=cfg.FPN.H,
            W=cfg.FPN.W,

            num_down_blocks=cfg.FPN.num_down_blocks,
            num_up_blocks=cfg.FPN.num_up_blocks,

            fu_mode=cfg.FPN.fu_mode,
            upsample_mode=cfg.FPN.upsample_mode,
            detector=cfg.FPN.detector,

            scale=cfg.FPN.scale,
            planes=cfg.FPN.planes
        )
    elif cfg.model == 'SUNet':
        model = SUNet_model(cfg)
    elif cfg.model == 'CTformer':
        model = CTformer_model(cfg)
    elif cfg.model == 'SwinIR':
        model = SwinIR_model(cfg)
    else:
        raise Exception('Unrecognized model name')
    
    return model


def build_optimizer(cfg, model):
    assert cfg.optim in ['SGD', 'Adam', 'AdamW'], 'Unrecognized optimizer name'
    if cfg.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)

    return optimizer


def build_scheduler(cfg, optimizer):
    assert cfg.scheduler in ['StepLR', 'Cosine'], 'Unrecognized scheduler name'
    if cfg.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.5)      # gmma=0.5
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    return scheduler


def build_loss(cfg):
    assert cfg.loss in ['MSE', 'L1', 'Huber'], 'Unrecognized loss name'
    if cfg.loss == 'MSE':
        loss_func = torch.nn.MSELoss()
    elif cfg.loss == 'L1':
        loss_func = torch.nn.L1Loss()
    elif cfg.loss == 'Huber':
        loss_func = torch.nn.HuberLoss()
    return loss_func


def save_checkpoint(model, optimizer, save_path, epoch, loss):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss }, save_path)


def load_checkpoint(model, optimizer, load_path):
   checkpoint = torch.load(load_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   epoch = checkpoint['epoch']
   loss = checkpoint['loss']
   return model, optimizer, epoch, loss


def model_size(model):
    param_size,  buffer_size = 0, 0
    for param in model.parameters():     # an iterator over params
        param_size += param.numel() * param.element_size()
    for buffer in model.buffers():  # an iterator over module buffers
        buffer_size += buffer.numel() * buffer.element_size()
        
    total_params_mb = (param_size + buffer_size) / 1024**2     
    print('model size: {:.3f}MB'.format(total_params_mb))