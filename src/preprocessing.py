import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import librosa
import multiprocessing as mp


NEGATIVE_CLASS = {"f2", "f3", "f4", "f5", "f6", "f9", "f10",
                  "m1", "m2", "m4", "m5", "m7", "m9", "m10"}
POSITIVE_CLASS = {"f1", "f7", "f8",
                  "m3", "m6", "m8"}


def overlapping_frames(x: np.ndarray, frame_size: int, overlap_factor: float) -> np.ndarray:
    assert len(x) >= frame_size, f"Frame size ({frame_size}) exceeds length of x ({len(x)})."
    assert overlap_factor >= 0.0 and overlap_factor < 1.0,\
        f"Overlap factor ({overlap_factor}) has to be from range [0.0, 1.0)."

    n = x.shape[0]
    stride_len = x.strides[0]

    stride_size = int(frame_size - np.floor(frame_size * overlap_factor))
    window_count = int(np.floor((n - frame_size)/stride_size)) + 1

    frames = stride_tricks.as_strided(x, shape=(window_count, frame_size),
                                      strides=(stride_len*stride_size, stride_len))

    return frames



def overlapping_frames2D(X: np.ndarray, frame_size: int, overlap_factor: float) -> np.ndarray:
    assert len(X.shape) == 2, "X is not a 2D matrix."
    height, width = X.shape
    assert width >= frame_size, f"Frame size ({frame_size}) exceeds length of x ({width})."

    assert overlap_factor >= 0.0 and overlap_factor < 1.0,\
        f"Overlap factor ({overlap_factor}) has to be from range [0.0, 1.0)."

    stride_size = int(frame_size - np.floor(frame_size * overlap_factor))
    # Remove offset from the beginning of the data, its just background noise
    offset = (width - frame_size) % stride_size
    X = X[:, offset:]
    height, width = X.shape
    window_count = (width - frame_size) // stride_size + 1

    stride_height, stride_width = X.strides
    frames = stride_tricks.as_strided(X, shape=(window_count, height, frame_size),
                                      strides=(stride_width*stride_size, stride_height, stride_width))

    return frames



def process_file(
        filename: str,
        input_path: str,
        output_path: dict[str, str],
        augmentations: dict[str]
):
    sr = None
    if "sr" in augmentations:
        sr = float(augmentations["sr"])

    offset = 0.0
    if "offset" in augmentations:
        offset = float(augmentations["offset"])

    n_mels = 256
    if "target_height" in augmentations:
        n_mels = int(augmentations["target_height"])

    filepath = os.path.join(input_path, filename)+".wav"
    print(f"[LOG] \tOpening file: {filepath}")

    y, sr = librosa.load(filepath, sr=sr, offset=offset)

    # Split y into frames (default size is len(y), producing 1 frame), which share some amount of
    # data equal to OVERLAP_FACTOR. One image of width duration*sample_rate would be too big for
    # training CNN, also model which requires ~3min sample to classify a person is not useful.
    if "overlap_factor" in augmentations:
        overlap_factor = float(augmentations["overlap_factor"])
    else:
        overlap_factor = 0.0


    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logS = librosa.power_to_db(S)

    height, width = logS.shape

    if "target_width" in augmentations:
        target_width = augmentations["target_width"]
        if type(target_width) is int:
            frame_size = target_width
        else:
            frame_size = int(width * (float(target_width) * sr / len(y)))
    else:
        frame_size = width

    frames = overlapping_frames2D(logS, frame_size, overlap_factor)

    # Standardize colormap
    vmin = np.min(logS)
    vmax = np.max(logS)

    if "test_split" in augmentations:
        test_split = float(augmentations["test_split"])
    else:
        test_split = 0.0

    idx = np.arange(len(frames))
    if "shuffle" in augmentations:
        shuffle = bool(augmentations["shuffle"])
        if shuffle:
            np.random.shuffle(idx)
            frames = frames[idx]

    train_frames_count = int(np.ceil((1.0 - test_split) * len(frames)))
    splited_idx = np.split(idx, [train_frames_count], axis=0)
    splited_frames = np.split(frames, [train_frames_count], axis=0)
    splited_base_output_paths = [os.path.join(output_path["train"], filename),
                                 os.path.join(output_path["test"], filename)]

    for idx, frames, base_output_path in zip(splited_idx, splited_frames, splited_base_output_paths):
        for i, frame in zip(idx, frames):
            output_file_path = base_output_path+f"_{i}.png"
            plt.imsave(output_file_path, frame, cmap="bwr", vmin=vmin, vmax=vmax)



def preprocess_dir(input_path: str, output_path: str, dir: str, augmentations: dict[str]):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    class_paths = { "0": {"train": os.path.join(output_path, "train", "0"),
                          "test": os.path.join(output_path, "test", "0")},
                    "1": {"train": os.path.join(output_path, "train", "1"),
                          "test": os.path.join(output_path, "test", "1")}}

    print(f"[LOG] Processing directory '{dir}'")
    full_input_path = os.path.join(input_path, dir)
    for entry in os.scandir(full_input_path):
        print(f"[LOG] \tProcessing entry '{entry.path}'")
        if entry.is_dir():
            print("[LOG] \tEntry is a directory, skipping")
            continue

        file_no_ext = entry.name.removesuffix(".wav")
        actor, script, *env = file_no_ext.split("_")
        env = "_".join(env)
        class_path = class_paths["1" if actor in POSITIVE_CLASS else "0"]

        cur_path = {}
        cur_path["train"] = os.path.join(class_path["train"], env, actor)
        os.makedirs(cur_path["train"], exist_ok=True)
        cur_path["test"] = os.path.join(class_path["test"], env, actor)
        os.makedirs(cur_path["test"], exist_ok=True)

        process_file(file_no_ext, full_input_path, cur_path, augmentations)
        print("[LOG] \tEntry processed")



def main():
    import argparse
    import time
    import json

    # TODO: Create some usage
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    data_path = config["data_path"]
    daps_path = os.path.join(data_path, "daps")
    output_path = os.path.join(data_path, config["output_dir"])
    dirs = config["dirs"]
    augmentations = config["augmentations"]

    pool = mp.Pool(4)
    print("START PROCESSING")
    start = time.time()
    results = [pool.apply_async(preprocess_dir,
                                [daps_path, output_path, dir, augmentations])
               for dir in dirs]

    for r in results:
        r.get()

    end = time.time()
    print("END PROCESSING")
    print(f"Elapsed time: {end - start} s")


if __name__ == "__main__":
    main()