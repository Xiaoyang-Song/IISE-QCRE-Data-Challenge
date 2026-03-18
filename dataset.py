import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATE_DICT = {
    "1": ["03", "04", "05"],
    "2": ["06", "07"],
    "3": ["08", "09", "12"],
    "4": ["13", "14", "15"],
    "5": ["16", "17", "18", "19"]
}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SplitUnlabeledImageDataset(Dataset):
    def __init__(
        self,
        data_root="Data",
        roots=None,
        image_size=(224, 224),
        convert_rgb=True,
        transform=None,
        return_metadata=False,
    ):
        """
        Each original image is split vertically into 2 halves:
        - left half
        - right half

        Each half becomes one dataset sample.

        Parameters
        ----------
        data_root : str
            Root data directory
        roots : list[int] or None
            Which Train Unlabeled folders to use, e.g. [1], [1,2,3]
        image_size : tuple
            Resize each half to this size, default (224, 224)
        convert_rgb : bool
            Convert image to RGB
        transform : torchvision transform or None
            If None, use default resize + ToTensor
        return_metadata : bool
            Whether to also return path/root/date/side
        """
        if roots is None:
            roots = [1]

        self.data_root = data_root
        self.roots = [str(r) for r in roots]
        self.image_size = image_size
        self.convert_rgb = convert_rgb
        self.return_metadata = return_metadata

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),   # [C, H, W], float in [0,1]
            ])
        else:
            self.transform = transform

        self.original_images = []
        self.samples = []

        self._build_index()

    def _build_index(self):
        grand_total_images = 0
        grand_total_samples = 0

        for root_idx in self.roots:
            root_name = f"Train Unlabeled 0{root_idx}"
            root_dir = os.path.join(self.data_root, root_name)

            if not os.path.isdir(root_dir):
                print(f"[Warning] Missing root folder: {root_dir}")
                continue

            root_image_count = 0
            print(f"\n=== Scanning {root_dir} ===")

            for day in DATE_DICT[root_idx]:
                date_name = f"2026-02-{day}"
                date_dir = os.path.join(root_dir, date_name)

                if not os.path.isdir(date_dir):
                    print(f"[Warning] Missing date folder: {date_dir}")
                    continue

                image_paths = sorted([
                    os.path.join(date_dir, f)
                    for f in os.listdir(date_dir)
                    if os.path.isfile(os.path.join(date_dir, f))
                    and os.path.splitext(f)[1].lower() in VALID_EXTS
                ])

                folder_count = len(image_paths)
                root_image_count += folder_count
                grand_total_images += folder_count

                print(f"[Folder] {date_dir} -> {folder_count} image(s)")

                for path in image_paths:
                    base_info = {
                        "path": path,
                        "root_idx": int(root_idx),
                        "root_name": root_name,
                        "date": date_name,
                    }
                    self.original_images.append(base_info)

                    self.samples.append({**base_info, "side": "left"})
                    self.samples.append({**base_info, "side": "right"})
                    grand_total_samples += 2

            print(f"[Root Summary] {root_dir} -> {root_image_count} original image(s)")

        print(f"\n[Grand Summary] Total original images indexed: {grand_total_images}")
        print(f"[Grand Summary] Total split samples indexed: {grand_total_samples}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        path = info["path"]
        side = info["side"]

        img = Image.open(path)
        if self.convert_rgb:
            img = img.convert("RGB")

        w, h = img.size
        mid = w // 2

        if side == "left":
            cropped = img.crop((0, 0, mid, h))
        else:
            cropped = img.crop((mid, 0, w, h))

        x = self.transform(cropped)

        if self.return_metadata:
            return x, info
        return x


def build_split_unlabeled_dataloader(
    data_root="Data",
    roots=None,
    image_size=(224, 224),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    convert_rgb=True,
    transform=None,
    return_metadata=False,
):
    dataset = SplitUnlabeledImageDataset(
        data_root=data_root,
        roots=roots,
        image_size=image_size,
        convert_rgb=convert_rgb,
        transform=transform,
        return_metadata=return_metadata,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader