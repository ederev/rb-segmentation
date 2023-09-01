import os
from pathlib import Path
from collections import OrderedDict
import segmentation_models_pytorch as smp
import torch
import pandas as pd
from segmentation.utils import config_parser
import argparse
from segmentation.models.conversion import export_trained_model, check_traced_model
from segmentation.data.datamodule import DataModule
from segmentation.models.model import Model

from segmentation.test import testset_evaluation
from segmentation.models.unet_custom import UNetCustom, down_block, up_block
import pytorch_lightning as pl
import tensorboard


def get_args():
    """Parsing command line arguments

    Returns:
        Namespace: dict of parsed command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_path", type=Path, required=True, help="project path directory"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="experiment config path"
    )
    return parser.parse_args()


def main(args):
    # get params from experiment config
    cfg = config_parser(args.config)
    cfg.project_path = args.project_path

    EXPERIMENT_PATH = Path(
        cfg.project_path, f"{cfg.model.arch_name}_{cfg.model.encoder_name}"
    )
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    """ Data Module """
    data_module = DataModule(
        dataset=cfg.data.dataset,
        data_dir=cfg.data.data_path,
        batch_size=cfg.data_loader.batch_size,
        num_workers=cfg.data_loader.num_workers,
        image_size=(
            cfg.data_loader.image_size.height,
            cfg.data_loader.image_size.width,
        ),
    )

    """ Model settings """
    # define model
    if cfg.model.arch_name == "UNetCustom":
        model = UNetCustom(
            down_block(),
            up_block(),
            min_channels=32,
            max_channels=256,
            depth=5,
            n_classes=cfg.model.num_classes,
        )
    else:
        model = smp.create_model(
            cfg.model.arch_name,
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=3,
            classes=cfg.model.num_classes,
        )

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # model
    ligtning_model = Model(
        model,
        optimizer,
        criterion,
        class_names=data_module.dataset_classnames,
        image_size=(
            cfg.data_loader.image_size.height,
            cfg.data_loader.image_size.width,
        ),
    )

    """ Trainer settings """
    # define callbacks and storage for checkpoints
    checkpoint_dir = Path(EXPERIMENT_PATH, "checkpoints")
    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{cfg.model.arch_name}_{cfg.model.encoder_name}",
        verbose=True,
        save_top_k=-1,
        monitor="valid_loss",
        mode="min",
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=15
    )
    callbacks = [
        checkpoint_callback,
        lr_monitor_callback,
        # early_stopping
    ]

    # define logger
    log_dir = Path(EXPERIMENT_PATH, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir, name=f"{cfg.model.arch_name}_{cfg.model.encoder_name}"
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=cfg.trainer.num_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )

    """ --- TRAINING --- """
    # run training
    trainer.fit(ligtning_model, data_module)

    """ --- TESTING --- """
    # run validation dataset
    valid_metrics = trainer.validate(ligtning_model, data_module, verbose=False)
    print("valid_metrics:", valid_metrics)
    df_valid_metrics = pd.DataFrame(valid_metrics)
    df_valid_metrics.to_csv(f"{EXPERIMENT_PATH}/valid_metrics.csv")

    # run test dataset
    test_metrics = trainer.test(ligtning_model, data_module, verbose=False)
    print("test_metrics:", test_metrics)
    df_test_metrics = pd.DataFrame(test_metrics)
    df_test_metrics.to_csv(f"{EXPERIMENT_PATH}/test_metrics.csv")

    """ --- SAVE & CONVERT TRAINED MODELS --- """
    print(
        f"checkpoint_callback.best_model_path --- {checkpoint_callback.best_model_path}"
    )
    state_dict = torch.load(checkpoint_callback.best_model_path)["state_dict"]
    lightning_state_dict = OrderedDict(
        [(key[6:], state_dict[key]) for key in state_dict.keys()]
    )

    model.load_state_dict(lightning_state_dict)
    model.eval()

    export_trained_model(
        model,
        experiment_path=EXPERIMENT_PATH,
        input_size=(
            cfg.data_loader.image_size.height,
            cfg.data_loader.image_size.width,
        ),
    )

    check_traced_model(
        img_path="data/test_image.png",
        model_path=Path(EXPERIMENT_PATH, "model_traced_trained.pt"),
        experiment_path=EXPERIMENT_PATH,
        input_size=(
            cfg.data_loader.image_size.height,
            cfg.data_loader.image_size.width,
        ),
    )

    # [optional final check] run extended test on inference model
    # model = torch.jit.load(Path(EXPERIMENT_PATH, "model_traced_trained.pt"))  # tmp check for debug
    testset_evaluation(
        inference_model=model,
        test_dataloader=data_module.test_dataloader(),
        criterion=criterion,
        num_classes=cfg.model.num_classes,
        dataset_labels=data_module.dataset_classnames,
        dataset_colors=data_module.dataset_colormap,
        experiment_path=EXPERIMENT_PATH,
    )


if __name__ == "__main__":
    # --project_path="/content/drive/MyDrive/DATA-SCIENCE/lyft-udacity-challenge/"
    # --config="../configs/unet.yaml"

    arguments = get_args()
    main(arguments)


# %load_ext tensorboard
# %tensorboard --logdir /content/drive/MyDrive/DATA-SCIENCE/lyft-udacity-challenge/UNet_timm-mobilenetv3_large_100/logs/
