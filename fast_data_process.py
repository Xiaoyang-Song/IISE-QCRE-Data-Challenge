import os
import argparse
from PIL import Image

import torch
from torchvision import transforms
from tqdm import tqdm

DATE_DICT = {
    "1": ["03", "04", "05"],
    "2": ["06", "07"],
    "3": ["08", "09", "12"],
    "4": ["13", "14", "15"],
    "5": ["16", "17", "18", "19"]
}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # [C, H, W], float32 in [0, 1]
])


def get_image_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in VALID_EXTS
    ])


def process_one_image(path, convert_rgb=True, use_fp16=False):
    img = Image.open(path)
    if convert_rgb:
        img = img.convert("RGB")

    w, h = img.size
    mid = w // 2

    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))

    left_tensor = TRANSFORM(left)
    right_tensor = TRANSFORM(right)

    if use_fp16:
        left_tensor = left_tensor.half()
        right_tensor = right_tensor.half()

    return left_tensor, right_tensor


def save_chunk(tensors, out_dir, base_name, chunk_idx):
    if len(tensors) == 0:
        return None

    X = torch.stack(tensors, dim=0)
    out_path = os.path.join(out_dir, f"{base_name}_chunk_{chunk_idx:03d}.pt")
    torch.save(X, out_path)

    size_gb = X.numel() * X.element_size() / 1e9
    print(f"[Saved] {out_path} | shape={tuple(X.shape)} | dtype={X.dtype} | approx={size_gb:.3f} GB")
    return out_path


def process_one_date(
    root_idx,
    day,
    data_root="Data",
    out_dir="Data/Processed_unlabeled",
    convert_rgb=True,
    use_fp16=False,
    chunk_size=512,
):
    root_name = f"Train Unlabeled 0{root_idx}"
    date_name = f"2026-02-{day}"
    folder = os.path.join(data_root, root_name, date_name)

    if not os.path.isdir(folder):
        print(f"[Warning] Missing folder: {folder}")
        return

    image_files = get_image_files(folder)
    n = len(image_files)

    print(f"\n[Folder] {folder}")
    print(f"[Info] Number of original images: {n}")

    if n == 0:
        print("[Skip] No images found")
        return

    os.makedirs(out_dir, exist_ok=True)

    base_name = f"X_unlabeled_root_{int(root_idx):02d}_{date_name}"
    bad_files = []

    buffer_tensors = []
    chunk_idx = 0
    total_saved = 0

    for i, path in enumerate(tqdm(image_files), start=1):
        try:
            left_tensor, right_tensor = process_one_image(
                path,
                convert_rgb=convert_rgb,
                use_fp16=use_fp16
            )

            buffer_tensors.append(left_tensor)
            buffer_tensors.append(right_tensor)

            if len(buffer_tensors) >= chunk_size:
                save_chunk(buffer_tensors, out_dir, base_name, chunk_idx)
                total_saved += len(buffer_tensors)
                chunk_idx += 1
                buffer_tensors = []

            if i % 100 == 0 or i == n:
                print(f"[Progress] {i}/{n} original images processed -> {2 * i} split images")

        except Exception as e:
            print(f"[Error] {path}: {e}")
            bad_files.append(path)

    if len(buffer_tensors) > 0:
        save_chunk(buffer_tensors, out_dir, base_name, chunk_idx)
        total_saved += len(buffer_tensors)

    print(f"[Done] {folder}")
    print(f"[Summary] Total split tensors saved: {total_saved}")

    if bad_files:
        bad_path = os.path.join(
            out_dir,
            f"bad_files_root_{int(root_idx):02d}_{date_name}.txt"
        )
        with open(bad_path, "w", encoding="utf-8") as f:
            for p in bad_files:
                f.write(p + "\n")
        print(f"[Warning] Logged {len(bad_files)} bad file(s) to {bad_path}")


def process_one_root(
    root_idx,
    data_root="Data",
    out_dir="Data/Processed_unlabeled",
    convert_rgb=True,
    use_fp16=False,
    chunk_size=512,
):
    root_total = 0
    print(f"\n========== Processing root {root_idx} ==========")

    for day in DATE_DICT[str(root_idx)]:
        folder = os.path.join(data_root, f"Train Unlabeled 0{root_idx}", f"2026-02-{day}")
        if os.path.isdir(folder):
            count = len(get_image_files(folder))
            print(f"[Count] {folder} -> {count} original image(s)")
            root_total += count
        else:
            print(f"[Warning] Missing folder: {folder}")

    print(f"[Root Summary] Root {root_idx} total original images: {root_total}")
    print(f"[Root Summary] Expected split images: {2 * root_total}")

    for day in DATE_DICT[str(root_idx)]:
        process_one_date(
            root_idx=root_idx,
            day=day,
            data_root=data_root,
            out_dir=out_dir,
            convert_rgb=convert_rgb,
            use_fp16=use_fp16,
            chunk_size=chunk_size,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder index, e.g. 1")
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--out_dir", default="Data/Processed_unlabeled")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Number of split images per saved .pt chunk")
    parser.add_argument("--fp16", action="store_true",
                        help="Save tensors as float16 instead of float32")
    args = parser.parse_args()

    process_one_root(
        root_idx=args.root,
        data_root=args.data_root,
        out_dir=args.out_dir,
        convert_rgb=True,
        use_fp16=args.fp16,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
    # python fast_data_process.py --root 1 --chunk_size 256 --fp16

    # Example load:
    # X = torch.load("Data/Processed_unlabeled/X_unlabeled_root_01_2026-02-03_chunk_000.pt")
    # print(X.shape)  # [N, 3, 224, 224]