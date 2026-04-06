from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as pl
import torch
import os

from Discriminator import LiteMultiModalDiscriminator
from SketchDataset import SketchDataset
from SketchEncoder import SketchEncoder
from Generator import Generator
from SketchGAN import SketchGAN

import warnings
warnings.filterwarnings('ignore')


torch.set_float32_matmul_precision("high")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = TensorBoardLogger("tb_logs", name="my_sketch_gan")


def main():

    # Dataset + Train/Val Split
    dataset = SketchDataset(root_dir="./Dataset/MM-CELEBA-HQ", image_size=256)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8 , pin_memory=True , persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True , persistent_workers=True)

 
    # Models
    sketch_enc = SketchEncoder()
    gen = Generator(text_dim=768)
    dis = LiteMultiModalDiscriminator(text_dim=768)

    model = SketchGAN(sketch_enc, gen, dis)

 
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="epoch-{epoch:02d}-fid-{fid:.2f}",
    monitor="fid",       
    mode="min",
    save_top_k=3,    
    save_last=True,
    every_n_epochs=1,   
    save_on_train_epoch_end=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="fid",
        patience=15,
        mode="min",
        verbose=True
    )

 
    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=80,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        benchmark=True,
        check_val_every_n_epoch=1
    )

    # Resume automatically
    last_ckpt = "checkpoints/last.ckpt"
    if os.path.exists(last_ckpt):
        print(f"Resuming from {last_ckpt}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)
    else:
        print("Starting fresh training...")
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
