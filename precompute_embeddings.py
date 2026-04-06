import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from ClipEncoder import ClipEncoder
from SketchDataset import SketchDataset


def precompute_clip_embeddings(root_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder = ClipEncoder().to(device)
    text_encoder.eval()

    save_dir = os.path.join(root_dir, 'text_embeddings')
    os.makedirs(save_dir, exist_ok=True)

    dataset = SketchDataset(root_dir=root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Starting CLIP precomputation on {device}...")

    with torch.no_grad():
        for batch_idx, (real_imgs, sketches, texts) in enumerate(tqdm(loader)):
            z_text = text_encoder(texts)  # Shape: (Batch, 512)

            embeddings = z_text.cpu()

            start_idx = batch_idx * batch_size
            for i in range(embeddings.size(0)):
                file_id = dataset.valid_file_ids[start_idx + i]
                torch.save(embeddings[i], os.path.join(save_dir, f"{file_id}.pt"))

    print(f"Finished! {len(dataset)} embeddings saved to {save_dir}")


if __name__ == "__main__":
    precompute_clip_embeddings("./Dataset/MM-CELEBA-HQ")