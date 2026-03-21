import os
import glob
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class CroppedToChunkedTensor:
    def __init__(
        self,
        root_dir="Data/cropped",
        save_dir="Data/cropped/chunked",
        image_size=224,
    ):
        self.root_dir = root_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),   # [3,224,224], float32 in [0,1]
        ])

    def _normalize_root_name(self, unlabeled_name: str):
        # Unlabeled_1 -> 01
        suffix = unlabeled_name.split("_")[-1]
        return f"{int(suffix):02d}"

    def _folder_matches_dates(self, folder_name, dates):
        """
        Accept either:
          folder_name = '2026-02-03_cropped'
        and dates can be:
          ['2026-02-03_cropped']  or ['2026-02-03']
        """
        if dates is None:
            return True

        for d in dates:
            if folder_name == d:
                return True
            if folder_name == f"{d}_cropped":
                return True
        return False

    def _extract_date_str(self, folder_name):
        """
        From '2026-02-03_cropped' -> '2026-02-03'
        If suffix not present, keep original.
        """
        if folder_name.endswith("_cropped"):
            return folder_name[:-8]
        return folder_name

    def _collect_image_paths(self, unlabeled_ids=None, dates=None):
        """
        Expect extracted folders like:
        Data/cropped/Unlabeled_1/2026-02-03_cropped/...images...
        """
        all_records = []

        if unlabeled_ids is None:
            unlabeled_ids = sorted([
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
                and d.lower().startswith("unlabeled_")
            ])

        for unlabeled_name in unlabeled_ids:
            unlabeled_path = os.path.join(self.root_dir, unlabeled_name)
            if not os.path.isdir(unlabeled_path):
                print(f"[Warning] Missing folder: {unlabeled_path}")
                continue

            date_dirs = sorted([
                d for d in os.listdir(unlabeled_path)
                if os.path.isdir(os.path.join(unlabeled_path, d))
            ])

            if dates is not None:
                date_dirs = [d for d in date_dirs if self._folder_matches_dates(d, dates)]

            for folder_name in date_dirs:
                folder_path = os.path.join(unlabeled_path, folder_name)
                date_str = self._extract_date_str(folder_name)

                for ext in VALID_EXTS:
                    for img_path in glob.glob(os.path.join(folder_path, "**", f"*{ext}"), recursive=True):
                        all_records.append((unlabeled_name, date_str, folder_name, img_path))
                    for img_path in glob.glob(os.path.join(folder_path, "**", f"*{ext.upper()}"), recursive=True):
                        all_records.append((unlabeled_name, date_str, folder_name, img_path))

        all_records = sorted(set(all_records))
        return all_records

    def save_chunks(
        self,
        unlabeled_ids=None,
        dates=None,
        chunk_size=2000,
        overwrite=False,
    ):
        """
        Save tensor chunks to:
            Data/cropped/chunked/X_unlabeled_root_01_2026-02-03_chunk_000.pt
        """
        records = self._collect_image_paths(unlabeled_ids=unlabeled_ids, dates=dates)

        if len(records) == 0:
            print("[Info] No images found.")
            return

        print(f"[Info] Found {len(records)} images total")

        # group by (unlabeled_name, date_str)
        groups = {}
        for unlabeled_name, date_str, folder_name, img_path in records:
            groups.setdefault((unlabeled_name, date_str), []).append(img_path)

        for (unlabeled_name, date_str), img_paths in sorted(groups.items()):
            root_id = self._normalize_root_name(unlabeled_name)
            img_paths = sorted(img_paths)

            print(f"\n[Processing] {unlabeled_name} | {date_str} | {len(img_paths)} images")

            chunk_tensors = []
            chunk_idx = 0
            num_skipped = 0

            for img_path in tqdm(img_paths, desc=f"{unlabeled_name}-{date_str}"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    x = self.transform(img)
                    chunk_tensors.append(x)
                except Exception as e:
                    num_skipped += 1
                    print(f"[Skip] {img_path} | {e}")
                    continue

                if len(chunk_tensors) >= chunk_size:
                    save_name = f"X_unlabeled_root_{root_id}_{date_str}_chunk_{chunk_idx:03d}.pt"
                    save_path = os.path.join(self.save_dir, save_name)

                    if (not overwrite) and os.path.exists(save_path):
                        print(f"[Exists] {save_path}")
                    else:
                        X = torch.stack(chunk_tensors, dim=0)
                        torch.save(X, save_path)
                        print(f"[Saved] {save_path} | shape={tuple(X.shape)}")

                    chunk_tensors = []
                    chunk_idx += 1

            if len(chunk_tensors) > 0:
                save_name = f"X_unlabeled_root_{root_id}_{date_str}_chunk_{chunk_idx:03d}.pt"
                save_path = os.path.join(self.save_dir, save_name)

                if (not overwrite) and os.path.exists(save_path):
                    print(f"[Exists] {save_path}")
                else:
                    X = torch.stack(chunk_tensors, dim=0)
                    torch.save(X, save_path)
                    print(f"[Saved] {save_path} | shape={tuple(X.shape)}")

            print(f"[Done] {unlabeled_name} | {date_str} | skipped={num_skipped}")


if __name__ == "__main__":
    maker = CroppedToChunkedTensor(
        root_dir="Data/cropped",
        save_dir="Data/cropped/chunked",
        image_size=224,
    )

    # everything
    # maker.save_chunks(
    #     unlabeled_ids=None,
    #     dates=None,
    #     chunk_size=2000,
    #     overwrite=False,
    # )

    # only one folder group
    # maker.save_chunks(
    #     unlabeled_ids=["Unlabeled_2", "Unlabeled_3"],
    #     dates=None,
    #     chunk_size=2000,
    # )

    # only one date, both forms work:
    # maker.save_chunks(
    #     unlabeled_ids=["Unlabeled_1"],
    #     dates=["2026-02-03"],
    #     chunk_size=2000,
    # )

    maker.save_chunks(
        unlabeled_ids=["Unlabeled_1"],
        dates=["2026-02-04_cropped", "2026-02-05_cropped"],
        chunk_size=2000,
    )