
import os
import time
from attrdict import AttrDict
import torch

from default_config import ModelTypes, mse_lpips_args
from src.helpers import datasets, utils


def eval(checkpoint_path: str, cpu: bool = False):
    
    cmd_args = AttrDict({
        "model_type": ModelTypes.COMPRESSION,
        "normalize_input_image": False,
        "save": "experiments",
        "use_latent_mixture_model": False,
        "warmstart": True,
        "warmstart_ckpt": checkpoint_path,
        "multigpu": False,
        "gpu": 0,
        "force_set_gpu": True,
        "n_residual_blocks": 9
    })

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    args = mse_lpips_args

    start_time = time.time()
    device = 'cpu' if cpu else utils.get_device()

    logger = utils.logger_setup(logpath=os.path.join(args.experiments, 'logs'), filepath=os.getcwd())

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)
    args.warmstart = cmd_args.warmstart
    args.warmstart_ckpt = cmd_args.warmstart_ckpt
    args.n_residual_blocks = 9

    assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
    # TODO : Define a custom model type and pass it through these frunctions
    args, model, _ = utils.load_model(args.warmstart_ckpt, logger, device, 
        model_type=args.model_type, current_args_d=dictify(args), strict=False, prediction=True)

    val_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                mode='validation',
                                shuffle=True,
                                normalize=args.normalize_input_image)

    for batch in val_loader:
        print(batch)
        outputs = model(batch[0])
        print(outputs)
        break
        

    print('Validation done.')

if __name__ == '__main__':
    eval(r'D:\Documents\project\thesis\experiments\hific_low.pt', True)