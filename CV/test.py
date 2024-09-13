import torch
from torchvision.transforms import v2
import pytorch_lightning as pl
from model import ImageClassifier
import warnings
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from types import SimpleNamespace
import yaml
import argparse


def get_resize_size(model_name):
    if model_name == "efficientnet_v2_s":
        return 384
    return 224


def get_config(config_path):
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    return config

def get_data_loader(test_dir, resize_size, batch_size=32):
    transform = v2.Compose([
        v2.Resize((resize_size, resize_size)),
        v2.CenterCrop(resize_size),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_dir, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=24)
    
    return test_loader

def main(args):
    config = SimpleNamespace(**get_config(args.config))
    model = ImageClassifier.load_from_checkpoint(config.model_weights_path)

    resize_size = get_resize_size(config.model_name)
    test_dataloader = get_data_loader(config.test_data_path, resize_size)

    trainer = pl.Trainer(
        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
        devices=(config.devices if torch.cuda.is_available() else "auto"),
        max_epochs=1,
        precision=16,
        deterministic=True,
    )

    trainer.test(model, dataloaders=test_dataloader)

if __name__ == "__main__":
    pl.seed_everything(42, workers=True) # same results

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    main(args)