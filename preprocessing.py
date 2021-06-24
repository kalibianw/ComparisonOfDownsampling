from utils import DataModule

dm = DataModule(data_dir_path="D:/AI/data/dogs-vs-cats/train/")

color_modes = ["rgb", "gray"]
for color_mode in color_modes:
    dm.img_to_npz(npz_path=f"npz/cats_dogs_{color_mode}.npz", color_mode=color_mode)
