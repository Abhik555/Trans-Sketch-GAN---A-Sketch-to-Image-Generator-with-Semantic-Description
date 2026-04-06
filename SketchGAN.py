# import torch
# import torch.nn.functional as F
# import lightning as pl
# from torch.optim import Adam
# import os
# import csv
# import torchvision
#
#
# class SketchGAN(pl.LightningModule):
#     def __init__(self, sketch_encoder, generator, discriminator, csv_path="training_log.csv"):
#         super().__init__()
#         # 1. Manual Optimization is mandatory for GANs with multiple optimizers
#         self.automatic_optimization = False
#
#         self.sketch_encoder = torch.compile(sketch_encoder,mode="reduce-overhead")
#         self.generator = torch.compile(generator, mode="reduce-overhead")
#
#         # 2. Compile Discriminator. If it still crashes, change mode to "default"
#         # or remove torch.compile temporarily to verify the rest of the logic.
#         self.discriminator = discriminator
#
#         self.csv_path = csv_path
#         self._init_csv()
#
#     def _init_csv(self):
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['epoch', 'batch_idx', 'g_loss', 'd_loss'])
#
#     def hinge_loss_dis(self, real_preds, fake_preds):
#         real_loss = torch.mean(F.relu(1.0 - real_preds))
#         fake_loss = torch.mean(F.relu(1.0 + fake_preds))
#         return real_loss + fake_loss
#
#     def hinge_loss_gen(self, fake_preds):
#         return -torch.mean(fake_preds)
#
#     def training_step(self, batch, batch_idx):
#         real_imgs, sketches, z_text = batch
#         opt_g, opt_d = self.optimizers()
#
#         # --- FORWARD PASS ---
#         # Generate once. We need the gradient chain for opt_g later.
#         z_sketch = self.sketch_encoder(sketches)
#         fake_imgs = self.generator(z_sketch, z_text)
#
#         # --- UPDATE DISCRIMINATOR (D) ---
#         self.toggle_optimizer(opt_d)
#
#         # IMPORTANT: We use .detach() on fake_imgs so D-update doesn't affect G-weights.
#         # This is the primary fix for the "inplace modified" error.
#         real_preds = self.discriminator(real_imgs, sketches, z_text)
#         fake_preds = self.discriminator(fake_imgs.detach(), sketches, z_text)
#
#         d_loss = self.hinge_loss_dis(real_preds, fake_preds)
#
#         opt_d.zero_grad()
#         self.manual_backward(d_loss)
#         opt_d.step()
#         self.untoggle_optimizer(opt_d)
#
#         # --- UPDATE GENERATOR (G) ---
#         self.toggle_optimizer(opt_g)
#
#         # Pass the original fake_imgs (with grad history) to get D's feedback
#         fake_preds_for_g = self.discriminator(fake_imgs, sketches, z_text)
#
#         g_adv_loss = self.hinge_loss_gen(fake_preds_for_g)
#         g_l1_loss = F.l1_loss(fake_imgs, real_imgs) * 10.0
#         g_loss = g_adv_loss + g_l1_loss
#
#         opt_g.zero_grad()
#         self.manual_backward(g_loss)
#         opt_g.step()
#         self.untoggle_optimizer(opt_g)
#
#         # --- LOGGING ---
#         # 1. Scalar Logs (TensorBoard)
#         self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)
#
#         # 2. Image Logs (TensorBoard) - Every 100 batches
#         if batch_idx % 100 == 0:
#             # Normalize [-1, 1] -> [0, 1] for display
#             # Grid: Sketch (replicated to 3ch) | Fake | Real
#             sketch_3ch = (sketches.repeat(1, 3, 1, 1) + 1) / 2
#             fake_vis = (fake_imgs + 1) / 2
#             real_vis = (real_imgs + 1) / 2
#
#             grid = torch.cat([sketch_3ch, fake_vis, real_vis], dim=-1)
#             self.logger.experiment.add_images('Training_Progress', grid, self.global_step)
#
#         # 3. CSV Logging (Continuous)
#         with open(self.csv_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([self.current_epoch, batch_idx, g_loss.item(), d_loss.item()])
#
#     def configure_optimizers(self):
#         # We must optimize the SketchEncoder as well!
#         g_params = list(self.generator.parameters()) + list(self.sketch_encoder.parameters())
#         opt_g = Adam(g_params, lr=1e-4, betas=(0.0, 0.9))
#         opt_d = Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
#         return [opt_g, opt_d], []

import torch
import torch.nn.functional as F
import lightning as pl
import torchvision
from torch.optim import Adam
from torchmetrics.image.fid import FrechetInceptionDistance
import os
import csv
import copy


class SketchGAN(pl.LightningModule):

    def __init__(
        self,
        sketch_encoder,
        generator,
        discriminator,
        csv_path="training_log.csv",
        accum_steps=2,              # manual gradient accumulation
        ema_decay=0.999
    ):
        super().__init__()

        # GANs require manual optimization
        self.automatic_optimization = False

        self.sketch_encoder = sketch_encoder
        self.generator = generator
        self.discriminator = discriminator

        # ----- Manual gradient accumulation -----
        self.accum_steps = accum_steps

        # ----- EMA Generator -----
        self.generator_ema = copy.deepcopy(self.generator)
        for p in self.generator_ema.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay

        # ----- FID Metric -----
        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=False
        )

        self.csv_path = csv_path
        self._init_csv()

    # =========================================================
    # Utilities
    # =========================================================

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'batch_idx', 'g_loss', 'd_loss'])

    def hinge_loss_dis(self, real_preds, fake_preds):
        real_loss = torch.mean(F.relu(1.0 - real_preds))
        fake_loss = torch.mean(F.relu(1.0 + fake_preds))
        return real_loss + fake_loss

    def hinge_loss_gen(self, fake_preds):
        return -torch.mean(fake_preds)

    # =========================================================
    # TRAINING
    # =========================================================

    def training_step(self, batch, batch_idx):
        real_imgs, sketches, z_text = batch
        opt_g, opt_d = self.optimizers()

        # Forward pass
        z_sketch = self.sketch_encoder(sketches)
        fake_imgs = self.generator(z_sketch, z_text)

        # =====================================================
        # DISCRIMINATOR UPDATE
        # =====================================================
        self.toggle_optimizer(opt_d)

        real_preds = self.discriminator(real_imgs, sketches, z_text)
        fake_preds = self.discriminator(fake_imgs.detach(), sketches, z_text)

        d_loss = self.hinge_loss_dis(real_preds, fake_preds)

        # Manual accumulation
        self.manual_backward(d_loss / self.accum_steps)

        if (batch_idx + 1) % self.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                1.0
            )
            opt_d.step()
            opt_d.zero_grad()

        self.untoggle_optimizer(opt_d)

        # =====================================================
        # GENERATOR UPDATE
        # =====================================================
        self.toggle_optimizer(opt_g)

        fake_preds_for_g = self.discriminator(fake_imgs, sketches, z_text)

        g_adv = self.hinge_loss_gen(fake_preds_for_g)
        g_l1 = F.l1_loss(fake_imgs, real_imgs) * 10.0
        g_loss = g_adv + g_l1

        self.manual_backward(g_loss / self.accum_steps)

        if (batch_idx + 1) % self.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.generator.parameters()) +
                list(self.sketch_encoder.parameters()),
                1.0
            )
            opt_g.step()
            opt_g.zero_grad()

        self.untoggle_optimizer(opt_g)

        # =====================================================
        # EMA UPDATE (only when generator steps)
        # =====================================================
        if (batch_idx + 1) % self.accum_steps == 0:
            with torch.no_grad():
                for p, p_ema in zip(
                    self.generator.parameters(),
                    self.generator_ema.parameters()
                ):
                    p_ema.data.mul_(self.ema_decay).add_(
                        p.data,
                        alpha=1 - self.ema_decay
                    )

        # Logging
        self.log_dict(
            {"g_loss": g_loss, "d_loss": d_loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        # =========================================================
        # IMAGE LOGGING (TensorBoard)
        # =========================================================
        if batch_idx % 200 == 0:  # log every 200 steps

            with torch.no_grad():
                fake_vis = self.generator_ema(z_sketch, z_text)

                # Limit number of samples to avoid VRAM spikes
                num_show = min(4, real_imgs.size(0))

                real_vis = real_imgs[:num_show]
                fake_vis = fake_vis[:num_show]
                sketch_vis = sketches[:num_show]

                # Convert grayscale sketch → 3 channels
                sketch_vis = sketch_vis.repeat(1, 3, 1, 1)

                # Normalize [-1,1] → [0,1]
                real_vis = (real_vis + 1) / 2
                fake_vis = (fake_vis + 1) / 2
                sketch_vis = (sketch_vis + 1) / 2

                # Concatenate horizontally: Sketch | Fake | Real
                combined = torch.cat(
                    [sketch_vis, fake_vis, real_vis],
                    dim=3
                )

                grid = torchvision.utils.make_grid(
                    combined,
                    nrow=1
                )

                self.logger.experiment.add_image(
                    "Training_Progress",
                    grid,
                    global_step=self.global_step
                )

        return g_loss

    # =========================================================
    # VALIDATION (FID)
    # =========================================================

    def on_validation_start(self):
        self.fid = self.fid.to(self.device)

    def validation_step(self, batch, batch_idx):
        real_imgs, sketches, z_text = batch

        z_sketch = self.sketch_encoder(sketches)
        fake_imgs = self.generator_ema(z_sketch, z_text)

        # Convert [-1,1] → [0,255] uint8
        real = ((real_imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
        fake = ((fake_imgs + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)

        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid.compute()

        self.log("fid", fid_score, prog_bar=True , on_epoch=True)

        print(f"\n🔥 FID Score: {fid_score:.4f}")

        self.fid.reset()

    # =========================================================
    # OPTIMIZERS
    # =========================================================

    def configure_optimizers(self):
        g_params = (
            list(self.generator.parameters()) +
            list(self.sketch_encoder.parameters())
        )

        opt_g = Adam(g_params, lr=1e-4, betas=(0.0, 0.9))
        opt_d = Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))

        return [opt_g, opt_d]
