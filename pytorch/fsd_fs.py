import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List

torch.manual_seed(42)
np.random.seed(42)


fsd50k_splits = {
    "dev_base": [
        "Chuckle_and_chortle",
        "Child_speech_and_kid_speaking",
        "Female_speech_and_woman_speaking",
        "Yell",
        "Gasp",
        "Giggle",
        "Conversation",
        "Male_speech_and_man_speaking",
        "Laughter",
        "Shout",
        "Walk_and_footsteps",
        "Finger_snapping",
        "Cheering",
        "Cough",
        "Chewing_and_mastication",
        "Breathing",
        "Chatter",
        "Clapping",
        "Whispering",
        "Sneeze",
        "Singing",
        "Burping_and_eructation",
        "Sigh",
        "Speech",
        "Purr",
        "Gull_and_seagull",
        "Meow",
        "Bird_vocalization_and_bird_call_and_bird_song",
        "Bark",
        "Crow",
        "Bird",
        "Frog",
        "Cat",
        "Dog",
        "Piano",
        "Cymbal",
        "Mallet_percussion",
        "Drum",
        "Trumpet",
        "Guitar",
        "Tambourine",
        "Gong",
        "Brass_instrument",
        "Percussion",
        "Keyboard_(musical)",
        "Scratching_(performance_technique)",
        "Harmonica",
        "Plucked_string_instrument",
        "Harp",
        "Raindrop",
        "Rain",
        "Ocean",
        "Stream",
        "Traffic_noise_and_roadway_noise",
        "Motorcycle",
        "Truck",
        "Computer_keyboard",
        "Typewriter",
        "Car",
        "Bus",
        "Skateboard",
        "Train",
        "Ringtone",
        "Bicycle",
        "Drill",
        "Sliding_door",
        "Slam",
        "Drawer_open_or_close",
        "Aircraft",
        "Frying_(food)",
        "Cupboard_open_or_close",
        "Clock",
        "Telephone",
        "Shatter",
        "Writing",
        "Zipper_(clothing)",
        "Boiling",
        "Typing",
        "Ratchet_and_pawl",
        "Power_tool",
        "Bathtub_(filling_or_washing)",
        "Hammer",
        "Boat_and_Water_vehicle",
        "Chink_and_clink",
        "Siren",
        "Microwave_oven",
        "Idling",
        "Door",
        "Accelerating_and_revving_and_vroom",
        "Packing_tape_and_duct_tape",
        "Fireworks",
        "Motor_vehicle_(road)",
        "Rail_transport",
        "Gunshot_and_gunfire",
        "Scissors",
        "Screech",
        "Crumpling_and_crinkling",
        "Tearing",
    ],
    "dev_val": [
        "Male_singing",
        "Speech_synthesizer",
        "Applause",
        "Fart",
        "Crowd",
        "Crying_and_sobbing",
        "Cricket",
        "Growling",
        "Insect",
        "Drum_kit",
        "Rattle_(instrument)",
        "Wind_instrument_and_woodwind_instrument",
        "Accordion",
        "Waves_and_surf",
        "Gurgling",
        "Trickle_and_dribble",
        "Subway_and_metro_and_underground",
        "Fixed-wing_aircraft_and_airplane",
        "Fill_(with_liquid)",
        "Engine_starting",
        "Splash_and_splatter",
        "Printer",
        "Keys_jangling",
        "Sink_(filling_or_washing)",
        "Mechanical_fan",
        "Cutlery_and_silverware",
        "Water_tap_and_faucet",
        "Pour",
        "Dishes_and_pots_and_pans",
        "Crushing",
    ],
    "eval": [
        "Female_singing",
        "Screaming",
        "Run",
        "Chicken_and_rooster",
        "Fowl",
        "Organ",
        "Bowed_string_instrument",
        "Thunder",
        "Tick-tock",
        "Sawing",
        "Toilet_flush",
        "Coin_(dropping)",
        "Boom",
        "Camera",
        "Drip",
    ],
}


class NaiveDataset:
    r"""An abstract dataset class."""

    def _load_audio(
        self, audio_path: str, sr: int = 44100, mono: bool = True
    ) -> torch.Tensor:
        wav, ori_sr = torchaudio.load(audio_path, normalize=True)
        if not ori_sr == sr:
            wav = torchaudio.functional.resample(wav, orig_freq=ori_sr, new_freq=sr)
        wav = wav.squeeze() if mono == True else wav
        return wav


class FSD_FS(NaiveDataset):
    r"""FSD-FS dataset."""
    clip_cfgs = {"sample_rate": 44100, "duration": 1, "hop_length": 0.5}

    def __init__(
        self,
        clip_dir: str,
        audio_dir: str,
        csv_path: str,
        *,
        mode: str = "base",
        data_type: str = "path",
        target_type: str = "category",
    ) -> None:
        super().__init__()
        self.cfgs = {"data_type": data_type, "target_type": target_type}
        self.clip_dir, self.csv_dir, self.audio_dir = (
            clip_dir,
            csv_path,
            os.path.join(audio_dir, mode),
        )
        # Prepare clip-level meta info
        self.meta = dict()
        os.makedirs(self.clip_dir, exist_ok=True)
        if len(os.listdir(self.clip_dir)) == 0:
            print("Start to curate audio clips.")
            csv_path = os.path.join(self.csv_dir, f"{mode}.csv")
            seg_meta = self._read_csv(csv_path)
            clip_data = self._make_clips(seg_meta)
            self._batch_dump(clip_data["clip_audio"], clip_data["clip_name"])
            # Get meta info and dump to csv file
            _str_cats, _fmt = list(), ","
            for name, cat in zip(clip_data["clip_name"], clip_data["clip_category"]):
                self.meta[name] = cat
                _str_cats.append(_fmt.join(cat))
            pd.DataFrame(
                {"file_name": clip_data["clip_name"], "category": _str_cats}
            ).to_csv(os.path.join(self.clip_dir, f"{mode}_clips.csv"), index=False)
        else:
            df = pd.read_csv(os.path.join(self.clip_dir, f"{mode}_clips.csv"))
            df["category"] = df["category"].apply(lambda x: str(x).split(","))
            for _, r in df.iterrows():
                self.meta[r["file_name"]] = r["category"]
        self.indices = list(self.meta.keys())

    def __len__(self):
        return len(self.indices)

    def __getitem__(
        self, item: Tuple[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns tensor of audio and its label."""
        x, y = item
        x = os.path.join(self.clip_dir, f"{x}.wav")
        if self.cfgs["data_type"] == "audio":
            x = torchaudio.load(x, normalize=True)
        if self.cfgs["target_type"] != "category":
            _tmp = list()
            _ys = y.split(", ")  # note: we assume separator is ', ' by default
            for _y in _ys:
                _tmp = self.tokeniser[self.cfgs["target_type"]](_y)
            y = _tmp
        return x, y

    def _read_csv(self, csv_path: str) -> dict:
        r"""Load meta information from a csv file.
        Returns a dict: 'file_name' -> (class) 'category'.
        """
        meta = dict()
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            meta[str(r["file_name"])] = r["category"].split(",")
        return meta

    def _make_clips(self, seg_meta: dict) -> dict:
        r"""Prepare audio clips with fixed length and
        returns clip-wise meta information: `clip_name` -> `clip_category`."""
        seg_names, seg_audios, seg_cats = list(), list(), list()
        for sname, scat in seg_meta.items():
            saudio, ori_sr = torchaudio.load(
                os.path.join(self.audio_dir, f"{sname}.wav"), normalize=True
            )
            seg_audios.append(saudio)
            seg_names.append(sname)
            seg_cats.append(scat)
        if ori_sr != FSD_FS.clip_cfgs["sample_rate"]:
            _tmp = list()
            resample_fn = torchaudio.transforms.Resample(ori_sr, sr)
            for s in seg_audios:
                _tmp.append(resample_fn(s))
            seg_audios = _tmp
        # Clip segments into the fixed length
        clip_duration = int(
            FSD_FS.clip_cfgs["sample_rate"] * FSD_FS.clip_cfgs["duration"]
        )
        clip_hop = int(clip_duration * FSD_FS.clip_cfgs["hop_length"])
        clip_names, clip_audios, clip_cats = list(), list(), list()
        for idx, seg in tqdm.tqdm(enumerate(seg_audios), total=len(seg_audios)):
            clips = self._clip_segment(
                wav=seg, win_length=clip_duration, hop_length=clip_hop
            )
            clip_audios.extend(clips)
            clip_names.extend(
                [seg_names[idx] + f"_{i:03d}" for i, _ in enumerate(clips)]
            )
            clip_cats.extend(seg_cats[idx] for _ in clips)
        assert len(clip_names) == len(clip_audios) == len(clip_cats)
        clip_data = {
            "clip_name": clip_names,
            "clip_audio": clip_audios,
            "clip_category": clip_cats,
        }
        return clip_data

    def _clip_segment(
        self, wav: torch.Tensor, win_length: int, hop_length: int
    ) -> List[torch.Tensor]:
        r"""Make variant-length a waveform into a fixed one by trancating and padding (replicating).
        wav is expected to be a channel_first tensor."""

        def _replicate(x: torch.Tensor, min_clip_duration: int) -> torch.Tensor:
            """Pad a 1-D tensor to fix-length `min_clip_duration` by replicating the existing elements."""
            tile_size = (min_clip_duration // x.size(dim=-1)) + 1
            x = torch.tile(x, dims=(tile_size,))[:min_clip_duration]
            return x

        clips = list()
        if wav.size(dim=-1) < win_length:
            tmp = _replicate(
                wav.squeeze(), win_length
            )  # transfer a mono 2-D waveform to 1-D tensor
            clips.append(
                tmp.unsqueeze(dim=0)
            )  # recover the waveform into size = (n_channel=1, n_samples)
        else:
            for idx in range(0, len(wav), hop_length):
                tmp = wav[idx : idx + win_length]
                tmp = _replicate(
                    tmp.squeeze(), win_length
                )  # to ensure the last seq have the same length
                clips.append(tmp.unsqueeze(dim=0))
        return clips

    def _batch_dump(self, audios: List[torch.Tensor], file_names: List[str]) -> None:
        r"""Dump a batch of audios separatively."""
        print(f"Store audio file(s) to {self.clip_dir}")
        for cname, caudio in zip(file_names, audios):
            torchaudio.save(
                filepath=os.path.join(self.clip_dir, f"{cname}.wav"),
                src=caudio,
                sample_rate=FSD_FS.clip_cfgs["sample_rate"],
                encoding="PCM_S",
            )

    def tokeniser(self) -> dict:
        r"""Returns a dict: 'class_category' -> dict(`class_id`, `class_mid`)"""
        csv_path = os.path.join(self.csv_dir, "vocabulary.csv")
        df = pd.read_csv(csv_path, names=["id", "category", "mid"])
        tokeniser = dict()
        for idx, r in df.iterrows():
            tokeniser[r["category"]] = {"id": r["id"], "mid": r["mid"]}
        return tokeniser

    @property
    def labelset(self):
        labels = list()
        for lset in self.meta.values():
            labels.extend(lset)
        return list(set(labels))


class MLFewShotSampler:
    r"""A multi-label few-shot sampler.
    Args:
        dataset: data source, e.g. instance of torch.data.Dataset, data[item] = {filename: labels}.
        n_class: number of novel classes in an episode (i.e., 'n' ways).
        n_supports: number of support samples in each novel class (i.e., 'k' shot).
        n_queries: number of queries for each novel class.
        n_task: total number of tasks (or episodes) in one epoch.
    """

    def __init__(
        self,
        dataset: torch.nn.Module,
        labelset: list,
        n_class: int = 15,
        n_supports: int = 5,
        n_queries: int = 5,
        n_task: int = 100,
        seperator: str = ", ",
        **kargs,
    ) -> None:
        self.dataset = dataset
        self.labelset = labelset
        self.n_cls = n_class
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.n_task = n_task
        self.seperator = seperator

    def __len__(self):
        return self.n_task

    def __iter__(self):
        r"""Returns a list of format: (`file_path`, `target_id`)."""
        for _ in range(self.n_task):
            batch_x, batch_y = list(), list()
            selected_classes = np.random.choice(
                self.labelset, size=self.n_cls, replace=False
            )
            # Create a data subset containing attached to novel classes ONLY
            subset = {"fpath": [], "label": []}
            for fpath, lbl in self.dataset.meta.items():
                if np.any(np.isin(lbl, selected_classes)):
                    subset["fpath"].append(fpath)
                    subset["label"].append(lbl)
            # Sample support examples and assign with labels
            for n in selected_classes:
                _candidate = list()
                for fpath, lbl in zip(subset["fpath"], subset["label"]):
                    if n in lbl:
                        _candidate.append(str(fpath))
                _samples = np.random.choice(
                    _candidate, size=self.n_supports, replace=False
                )
                batch_x.extend(_samples.tolist())
            supports = batch_x
            # Sample query examples and assign with labels
            for n in selected_classes:
                _candidate = list()
                for fpath, lbl in zip(subset["fpath"], subset["label"]):
                    if n in lbl and (fpath not in supports):
                        _candidate.append(str(fpath))
                _samples = np.random.choice(
                    _candidate, size=self.n_queries, replace=False
                )
                batch_x.extend(_samples.tolist())
            selected_classes = set(selected_classes)
            # Mask Unselected categories in the labels
            for x in batch_x:
                y = set(self.dataset.meta[x])
                y = list(y & selected_classes)
                batch_y.append(self.seperator.join(y))
            yield zip(batch_x, batch_y)


if __name__ == "__main__":
    clip_dir = "path/to/clip_dir"
    audio_dir = "path/to/audio_dir"
    csv_dir = "path/to/csv_dir"
    val = "dev_val"  # ["dev_base", "dev_val", "eval"]

    fsdfs = FSD_FS(
        clip_dir=clip_dir,
        audio_dir=audio_dir,
        csv_path=csv_dir,
        mode=val,
        data_type="path",
        target_type="category",
    )
    # tokeniser = fsdfs.tokeniser()
    # detokeniser = dict()
    # for cat, token in tokeniser.items():
    #     detokeniser[token["mid"]] = cat
    # selected_list = [detokeniser[mid] for mid in fsd50k_select_ids[val]]
    sampler = MLFewShotSampler(
        dataset=fsdfs,
        labelset=fsd50k_splits[val],
        n_class=15,
        n_supports=1,
        n_queries=5,
        n_task=100,
    )
    dataloader = DataLoader(
        fsdfs, batch_sampler=sampler, num_workers=4, pin_memory=True
    )
    for x, y in dataloader:
        print(f"x={x}\n")
        print(f"target={y}\n")
