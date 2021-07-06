from utils import DataModule

DATA_DIR_PATH = "D:/AI/data/A Large Scale Fish Dataset/Fish_Dataset/"
dm = DataModule(data_dir_path=DATA_DIR_PATH)

COLOR_MODES = ["rgb", "gray"]
NORMALIZED_STATUS = True
if NORMALIZED_STATUS is True:
    for color_mode in COLOR_MODES:
        dm.img_to_npz(npz_path=f"npz/normalized_fish_{color_mode}.npz",
                      color_mode=color_mode, normalized=NORMALIZED_STATUS)
elif NORMALIZED_STATUS is False:
    for color_mode in COLOR_MODES:
        dm.img_to_npz(npz_path=f"npz/fish_{color_mode}.npz", color_mode=color_mode, normalized=NORMALIZED_STATUS)
