import os
import argparse
import numpy as np
from PIL import Image

DATE_DICT = {
    "1": ["03", "04", "05"],
    "2": ["06", "07"],
    "3": ["08", "09", "12"],
    "4": ["13", "14", "15"],
    "5": ["16", "17", "18", "19"]
}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_image_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in VALID_EXTS
    ])


def process_one_date(root_idx, day, data_root="Data", out_dir="processed_unlabeled", convert_rgb=False):
    root_name = f"Train Unlabeled 0{root_idx}"
    date_name = f"2026-02-{day}"
    folder = os.path.join(data_root, root_name, date_name)

    if not os.path.isdir(folder):
        print(f"[Warning] Missing folder: {folder}")
        return

    image_files = get_image_files(folder)
    n = len(image_files)

    print(f"\n[Folder] {folder}")
    print(f"[Info] Number of images: {n}")

    if n == 0:
        print("[Skip] No images found")
        return

    os.makedirs(out_dir, exist_ok=True)

    first_img = Image.open(image_files[0])
    if convert_rgb:
        first_img = first_img.convert("RGB")
    first_arr = np.array(first_img)

    shape = first_arr.shape
    dtype = first_arr.dtype

    print(f"[Info] Image shape: {shape}")
    print(f"[Info] dtype: {dtype}")

    out_path = os.path.join(
        out_dir,
        f"X_unlabeled_root_{int(root_idx):02d}_{date_name}.npy"
    )

    # Use memmap so data is written to disk instead of staying in RAM
    X = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=dtype,
        shape=(n, *shape)
    )

    X[0] = first_arr

    bad_files = []

    for i, path in enumerate(image_files[1:], start=1):
        try:
            img = Image.open(path)
            if convert_rgb:
                img = img.convert("RGB")
            arr = np.array(img)

            if arr.shape != shape:
                raise ValueError(f"Shape mismatch: expected {shape}, got {arr.shape}")

            X[i] = arr

            if i % 100 == 0 or i == n - 1:
                print(f"[Progress] {i+1}/{n}")

        except Exception as e:
            print(f"[Error] {path}: {e}")
            bad_files.append(path)

    del X  # flush memmap to disk

    print(f"[Saved] {out_path}")

    if bad_files:
        bad_path = os.path.join(
            out_dir,
            f"bad_files_root_{int(root_idx):02d}_{date_name}.txt"
        )
        with open(bad_path, "w", encoding="utf-8") as f:
            for p in bad_files:
                f.write(p + "\n")
        print(f"[Warning] Logged {len(bad_files)} bad file(s) to {bad_path}")


def process_one_root(root_idx, data_root="Data", out_dir="processed_unlabeled", convert_rgb=False):
    root_total = 0
    print(f"\n========== Processing root {root_idx} ==========")

    for day in DATE_DICT[str(root_idx)]:
        folder = os.path.join(data_root, f"Train Unlabeled 0{root_idx}", f"2026-02-{day}")
        if os.path.isdir(folder):
            count = len(get_image_files(folder))
            print(f"[Count] {folder} -> {count} image(s)")
            root_total += count
        else:
            print(f"[Warning] Missing folder: {folder}")

    print(f"[Root Summary] Root {root_idx} total images: {root_total}")

    for day in DATE_DICT[str(root_idx)]:
        process_one_date(
            root_idx=root_idx,
            day=day,
            data_root=data_root,
            out_dir=out_dir,
            convert_rgb=convert_rgb
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder index, e.g. 1")
    parser.add_argument("--data_root", default="Data")
    parser.add_argument("--out_dir", default="Data/Processed_unlabeled")
    parser.add_argument("--rgb", action="store_true")

    args = parser.parse_args()

    process_one_root(
        root_idx=args.root,
        data_root=args.data_root,
        out_dir=args.out_dir,
        convert_rgb=args.rgb
    )


if __name__ == "__main__":
    main()

    # Load back
    # X = np.load("processed_unlabeled/X_unlabeled_root_01_2026-02-03.npy", mmap_mode="r")
    # print(X.shape)