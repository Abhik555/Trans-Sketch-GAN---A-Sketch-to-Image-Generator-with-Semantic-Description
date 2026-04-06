import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from ClipEncoder import ClipEncoder
from SketchDataset import SketchDataset


# Import your existing class here
# from SketchGAN import ClipEncoder

def precompute_clip_embeddings(root_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize YOUR ClipEncoder
    # We move it to device and set to eval mode to ensure it's frozen
    text_encoder = ClipEncoder().to(device)
    text_encoder.eval()

    # 2. Setup Save Directory
    save_dir = os.path.join(root_dir, 'text_embeddings')
    os.makedirs(save_dir, exist_ok=True)

    # 3. Load Dataset
    # Make sure your SketchDataset is the version that filters missing files
    dataset = SketchDataset(root_dir=root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"🚀 Starting CLIP precomputation on {device}...")

    with torch.no_grad():
        for batch_idx, (real_imgs, sketches, texts) in enumerate(tqdm(loader)):
            # texts is a tuple of strings from the batch
            # Your ClipEncoder handles tokenization internally
            z_text = text_encoder(texts)  # Shape: (Batch, 512)

            # Move to CPU before saving to keep GPU memory clean
            embeddings = z_text.cpu()

            # Save each file using the valid_file_ids from the dataset
            start_idx = batch_idx * batch_size
            for i in range(embeddings.size(0)):
                file_id = dataset.valid_file_ids[start_idx + i]
                torch.save(embeddings[i], os.path.join(save_dir, f"{file_id}.pt"))

    print(f"✅ Finished! {len(dataset)} embeddings saved to {save_dir}")


if __name__ == "__main__":
    precompute_clip_embeddings("./Dataset/MM-CELEBA-HQ")