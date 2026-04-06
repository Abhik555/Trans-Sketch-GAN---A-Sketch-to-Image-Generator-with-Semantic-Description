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

# Enable TF32 (very important for RTX 40 series)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = TensorBoardLogger("tb_logs", name="my_sketch_gan")


def main():

    # =========================
    # Dataset + Train/Val Split
    # =========================
    dataset = SketchDataset(root_dir="./Dataset/MM-CELEBA-HQ", image_size=256)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8 , pin_memory=True , persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True , persistent_workers=True)

    # =========================
    # Models
    # =========================
    sketch_enc = SketchEncoder()
    gen = Generator(text_dim=768)
    dis = LiteMultiModalDiscriminator(text_dim=768)

    model = SketchGAN(sketch_enc, gen, dis)

    # =========================
    # Callbacks
    # =========================

    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="epoch-{epoch:02d}-fid-{fid:.2f}",
    monitor="fid",          # still track best models
    mode="min",
    save_top_k=3,           # keep best 3 based on FID
    save_last=True,         # always save last.ckpt
    every_n_epochs=1,       # save every epoch
    save_on_train_epoch_end=True,  # ensure saving at end of epoch
    )

    early_stop_callback = EarlyStopping(
        monitor="fid",
        patience=15,
        mode="min",
        verbose=True
    )

    # =========================
    # Trainer
    # =========================
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
        print(f"🔄 Resuming from {last_ckpt}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)
    else:
        print("🚀 Starting fresh training...")
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
