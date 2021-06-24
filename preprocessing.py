from utils import DataModule

dm = DataModule(data_dir_path="D:/AI/data/dogs-vs-cats/train/")

COLOR_MODES = ["rgb", "gray"]
NORMALIZED_STATUS = True
if NORMALIZED_STATUS is True:
    for color_mode in COLOR_MODES:
        dm.img_to_npz(npz_path=f"npz/normalized_cats_dogs_{color_mode}.npz",
                      color_mode=color_mode, normalized=NORMALIZED_STATUS)
elif NORMALIZED_STATUS is False:
    for color_mode in COLOR_MODES:
        dm.img_to_npz(npz_path=f"npz/cats_dogs_{color_mode}.npz", color_mode=color_mode, normalized=NORMALIZED_STATUS)
