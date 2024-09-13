import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch
import torch.nn as nn
from torchvision import transforms
import pytorch_lightning as pl
from types import SimpleNamespace
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from model import ImageClassifier
import yaml
import warnings
import argparse

def get_resize_size(model_name):
    if model_name == "efficientnet_v2_s":
        return 384
    return 224


def get_config(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def get_data_loaders(train_dir, val_dir, resize_size, batch_size=32):
    train_transforms = v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.CenterCrop(resize_size),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=108),
        v2.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.CenterCrop(resize_size),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24)
    
    return train_loader, val_loader, train_dataset.classes

def main(args):
    config = SimpleNamespace(**get_config(args.config))
    model = ImageClassifier(
        class_size=config.class_size,
        eval_class_size=config.eval_class_size,
        lr=args.learning_rate
    )
    resize_size = get_resize_size(config.model_name)
    train_dataloader, val_dataloader, _ = get_data_loaders(config.train_data_path, config.eval_data_path, resize_size)

    profiler = PyTorchProfiler(
        schedule=torch.profiler.schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/last_profile"),
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    )

    logger = pl.loggers.CSVLogger(save_dir='logs/', name=config.model_name)
    tb_logger = TensorBoardLogger(save_dir="tb_logs/", name=config.model_name)

    ckpt = pl.callbacks.ModelCheckpoint(
        monitor='valid_f1',
        save_top_k=1,
        save_last=True,
        mode='max',
        auto_insert_metric_name=False,
        save_weights_only=True,
        filename='{epoch:02d}_{valid_f1:.3f}'
    )

    last_ckpt = pl.callbacks.ModelCheckpoint(
        monitor=None,
        auto_insert_metric_name=False,
        save_weights_only=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        strategy="ddp",
        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
        devices=(config.devices if torch.cuda.is_available() else "auto"),
        callbacks=[ckpt, last_ckpt, lr_monitor],
        logger=[logger, tb_logger],
        max_epochs=args.num_epochs,
        precision=16,
        # profiler=profiler,
        deterministic=True,
        accumulate_grad_batches=3, #speed up
        gradient_clip_val=1.0
    )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=int, default=5e-5)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)
