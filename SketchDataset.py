import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from concurrent.futures import ThreadPoolExecutor


class SketchDataset(Dataset):
    def __init__(self, root_dir, image_size=256, cache_file="dataset_cache.json"):
        self.root_dir = root_dir
        self.image_size = image_size
        self.cache_path = os.path.join(root_dir, cache_file)

        self.image_dir = os.path.join(root_dir, 'images')
        self.sketch_dir = os.path.join(root_dir, 'sketch', 'sketch')
        self.text_dir = os.path.join(root_dir, 'text', 'celeba-caption')
        self.emb_dir = os.path.join(root_dir, 'text_embeddings')

        # --- Fast Loading via Cache ---
        if os.path.exists(self.cache_path):
            print(f"📂 Loading dataset list from cache: {self.cache_path}")
            with open(self.cache_path, 'r') as f:
                self.valid_file_ids = json.load(f)
        else:
            self.valid_file_ids = self._scan_dataset_multithreaded()
            # Save cache for next time
            with open(self.cache_path, 'w') as f:
                json.dump(self.valid_file_ids, f)
            print(f"💾 Dataset cache saved to {self.cache_path}")

        # Standard Sort
        try:
            self.valid_file_ids.sort(key=lambda x: int(x))
        except ValueError:
            self.valid_file_ids.sort()

        # --- Transforms ---
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.sketch_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def _check_id(self, fid):
        """Worker function for threading: checks if all files exist for one ID"""
        # 1. Check Embedding (.pt)
        emb_exists = os.path.exists(os.path.join(self.emb_dir, f"{fid}.pt"))
        # 2. Check Image (jpg/png)
        img_exists = any(os.path.exists(os.path.join(self.image_dir, f"{fid}{ext}"))
                         for ext in ['.jpg', '.png', '.jpeg'])
        # 3. Check Sketch (jpg/png)
        skc_exists = any(os.path.exists(os.path.join(self.sketch_dir, f"{fid}{ext}"))
                         for ext in ['.jpg', '.png', '.jpeg'])

        if emb_exists and img_exists and skc_exists:
            return fid
        return None

    def _scan_dataset_multithreaded(self):
        print(f"🔍 Scanning dataset using multi-threading...")
        # Get all potential IDs from the embedding folder (since we precomputed them)
        all_potential_ids = [f.split('.')[0] for f in os.listdir(self.emb_dir) if f.endswith('.pt')]

        valid_ids = []
        # Use ThreadPoolExecutor (Multi-threading is better than Multi-processing for I/O checks)
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Map the worker function across all IDs
            results = list(executor.map(self._check_id, all_potential_ids))

        # Filter out the None results
        valid_ids = [r for r in results if r is not None]
        print(f"✅ Found {len(valid_ids)} complete samples.")
        return valid_ids

    def __len__(self):
        return len(self.valid_file_ids)

    def _get_image_path(self, directory, file_id):
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(directory, f"{file_id}{ext}")
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        file_id = self.valid_file_ids[idx]

        # Load precomputed embedding
        emb_path = os.path.join(self.emb_dir, f"{file_id}.pt")
        z_text = torch.load(emb_path, weights_only=True)

        # Load and transform Image
        img_path = self._get_image_path(self.image_dir, file_id)
        image = self.image_transform(Image.open(img_path).convert("RGB"))

        # Load and transform Sketch
        sketch_path = self._get_image_path(self.sketch_dir, file_id)
        sketch = self.sketch_transform(Image.open(sketch_path).convert("L"))

        return image, sketch, z_text