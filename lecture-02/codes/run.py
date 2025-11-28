import os
import yaml
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models import *
from experiment import VAEXperiment

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset import VAEDataset

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument(
        '--config',  '-c',
        dest="filename",
        metavar='FILE',
        help =  'path to the config file',
        default='configs/vae.yaml'
    )
    
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    tb_logger =  TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['model_params']['name'],
    )
    
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    
    # ----------------------
    # Device / accelerator logic
    # ----------------------
    trainer_params = config['trainer_params'].copy()

    # "gpus" can be an int (e.g., 1) or a list (e.g., [0, 1])
    raw_gpus = trainer_params.pop("gpus", 0)
    
    if isinstance(raw_gpus, (list, tuple)):
        num_gpus = len(raw_gpus)
        devices = list(raw_gpus)        # pass the list directly to Trainer
    else:
        num_gpus = int(raw_gpus) if raw_gpus else 0
        devices = num_gpus if num_gpus > 0 else 1   # will be overridden for CPU branch anyway
    
    if torch.cuda.is_available() and num_gpus > 0:
        accelerator = "gpu"
        strategy = DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else None
        cudnn.benchmark = True
        print(f"Using CUDA with {num_gpus} GPU(s): {devices}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) path
        accelerator = "mps"
        devices = 1
        strategy = None
        print("Using Apple MPS device.")
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None
        print("Using CPU.")
    
    # ----------------------
    # Model & experiment
    # ----------------------
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    
    # pin_memory only makes sense when using CUDA
    use_pin_memory = torch.cuda.is_available()
    data = VAEDataset(**config["data_params"], pin_memory=use_pin_memory)
    data.setup()
    
    # ----------------------
    # Trainer
    # ----------------------
    trainer_kwargs = dict(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True,
            ),
        ],
        accelerator=accelerator,
        devices=devices,
        **trainer_params,
    )
    
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    
    runner = Trainer(**trainer_kwargs)
    
    # ----------------------
    # Folders & run
    # ----------------------
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == "__main__":
    main()