import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import librosa
import time
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



def process_file(filename: str, input_path: str, output_path: str, augmentations: dict[str], **kwargs):
    sr = None
    if "sr" in kwargs:
        sr = float(kwargs["sr"])

    offset = 0.0
    if "offset" in kwargs:
        offset = float(kwargs["offset"])

    n_mels = 256
    if "n_mels" in kwargs:
        n_mels = int(kwargs["n_mels"])

    filepath = os.path.join(input_path, filename)+".wav"
    print(f"[LOG] \tOpening file: {filepath}")

    y, sr = librosa.load(filepath, sr=sr, offset=offset)

    # if "mute_sections" in augmentations and augmentations["mute_sections"]:
    #     intervals = librosa.effects.split(y=y,)

    # Split y into frames (default size is len(y), producing 1 frame), which share some amount of
    # data equal to OVERLAP_FACTOR. One image of width duration*sample_rate would be too big for
    # training CNN, also model which requires ~3min sample to classify a person is not useful.
    if "frame_size" in augmentations:
        frame_size = int(augmentations["frame_size"] * sr)
    else:
        frame_size = len(y)

    if "overlap_factor" in augmentations:
        overlap_factor = float(augmentations["overlap_factor"])
    else:
        overlap_factor = 0.0


    frames = overlapping_frames(y, frame_size, overlap_factor)

    S = [librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=n_mels) for frame in frames]
    logS = [librosa.power_to_db(s) for s in S]


    # Standardize colormap
    vmin = np.min(logS)
    vmax = np.max(logS)

    base_output_path = os.path.join(output_path, filename)
    for i, logs in enumerate(logS):
        output_file_path = base_output_path+f"_{i}.png"
        plt.imsave(output_file_path, logs, cmap="bwr", vmin=vmin, vmax=vmax)
        print(f"[LOG] \tSaved file: {output_file_path}")



def preprocess_dir(data_path: str, output_path: str, dir: str, augmentations: dict[str], **kwargs):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    class_paths = { "0": os.path.join(output_path, "0"),
                    "1": os.path.join(output_path, "1")}

    for class_path in class_paths.values():
        if not os.path.exists(class_path):
            os.mkdir(class_path)


    print(f"[LOG] Processing directory '{dir}'")
    full_dir_path = os.path.join(data_path, dir)
    for entry in os.scandir(full_dir_path):
        print(f"[LOG] \tProcessing entry '{entry.path}'")
        if entry.is_dir():
            print("[LOG] \tEntry is a directory, skipping")
            continue

        file_no_ext = entry.name.removesuffix(".wav")
        actor, *_ = file_no_ext.split("_")
        class_path = class_paths["1" if actor in POSITIVE_CLASS else "0"]
        process_file(file_no_ext, full_dir_path, class_path, augmentations, **kwargs)

        print("[LOG] \tEntry processed")



def main():
    data_path = os.path.join(os.getcwd(), "..", "data")
    daps_path = os.path.join(data_path, "daps")
    output_path = os.path.join(data_path, "daps_img")
    dirs = ["iphone_bedroom1", "ipad_confroom2", "ipadflat_office1", "ipadflat_confroom1",
            "ipad_balcony1", "clean", "produced", "cleanraw", "ipad_bedroom1", "ipad_livingroom1",
            "ipad_confroom1", "ipad_office1", "ipad_office2", "iphone_balcony1",
            "iphone_livingroom1"]
    augmentations = {"mute_sections": False,
                     "overlap_factor": 0.33,
                     "frame_size": 3.0}

    pool = mp.Pool(4)
    print("START PROCESSING")
    start = time.time()
    results = [pool.apply_async(preprocess_dir,
                                [daps_path, output_path, dir, augmentations])
               for dir in dirs]
    for r in results:
        r.wait()
    end = time.time()
    print("END PROCESSING")
    print(f"Elapsed time: {end - start} s")


if __name__ == "__main__":
    main()