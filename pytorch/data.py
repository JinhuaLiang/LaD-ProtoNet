import os
import time
import pickle
import csv
import torch
import torchaudio
import tqdm
import numpy as np
import pandas as pd

import pickle as pkl

from abc import abstractmethod
from typing import Tuple, Optional, Union, List, Callable, Any
from torch import Tensor
from torch.nn.functional import one_hot

from pytorch.utils import fix_len, load_wav, create_task_label
from pytorch.embed_label import AudioSetTaxonomy, label_embedding
from src.manifolds import set_logger, make_folder, float32_to_int16, int16_to_float32
from src.esc_meta import esc_hierarchy
from src.fsd_meta import FSD50K_MetaContainer

# Set the backend for audio processing
torchaudio.set_audio_backend("sox_io")


class NaiveDataset(object):
    """" An abstract dataset class"""
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class ESC50(NaiveDataset):
    """ ESC-50 dataset."""
    def __init__(
            self,
            wav_dir: str,
            csv_path: str,
            fold: list,
            num_class: int = 50,
            sample_rate: int = 44100
    ) -> None:
        self.wav_dir = wav_dir
        self.num_class = num_class
        self.sample_rate = sample_rate
        self.meta = self._load_meta(csv_path, fold)  # {filename: child_id}
        assert len(self.meta) > 0
        self.pid = self._build_hierarchy(csv_path)  # {id: pid}
        self.indices = list(self.meta.keys())  # create indices for filename

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: Union[str, int, tuple]) -> Tuple[Tensor, Tensor]:
        if isinstance(item, tuple):
            fname, label = item
            if not isinstance(label, Tensor):
                label = torch.tensor(label)
        else:
            fname = self.indices[item] if isinstance(item, int) else item
            label = torch.tensor(self.meta[fname])
        return load_wav(os.path.join(self.wav_dir, fname), sr=self.sample_rate), label

    def _load_meta(self, csv_path: str, fold: list) -> dict:
        """ Load meta info in line with cross-validation
            Args:
                csv_path: str, path to esc50.csv
                fold: list, fold id(s) needed for train/val dataset
        """
        meta = {}
        with open(csv_path, 'r') as f:  # esc50.csv organise the meta in the terms of
            rows = csv.DictReader(f)  # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
            for r in rows:
                if int(r['fold']) in fold:
                    meta[r['filename']] = int(r['target'])  # convert to int idx from str type from csv file

        return meta  # format = {filename: (child_id, parent_id)}

    def _build_hierarchy(self, csv_path: str):
        _, _, vocabulary = self.create_vocabulary(csv_path)
        map = dict()  # {cid: pid}
        for value in vocabulary.values():
            map[value['id']] = value['pid']

        return map

    @classmethod
    def create_vocabulary(cls, csv_path: str):
        """ Create vocabulary sets needs for few-shot learning.
        :param csv_path: path/to/dir/meta/esc50.csv
        :return: voc = {'class_name': class_id},
                 parent_voc = {'parent_class_name': parent_class_id}
                 vocabulary = {'class_name': {'pid', 'id'}}
        """
        voc = dict()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for d in reader:
                if d['target'] not in voc.keys():
                    voc[d['category']] = int(d['target'])

        parent_voc = dict()
        for idx, item in enumerate(esc_hierarchy.keys()):
            parent_voc[item] = idx

        vocabulary = dict()
        for key, value in voc.items():
            for p_cat in parent_voc.keys():
                if key in esc_hierarchy[p_cat]:
                    pid = parent_voc[p_cat]
                vocabulary[key] = {'pid': pid, 'id': value}
        return voc, parent_voc, vocabulary


# class FSD_MIX(NaiveDataset):
#     def __init__(self, pkl_dir, split, mode=None):
#         if split == 'base':
#             assert mode in ['train', 'test', 'val']
#             pkl_path = os.path.join(pkl_dir, f"{split}_{mode}.pkl")
#         elif split in ['test', 'val']:
#             pkl_path = os.path.join(pkl_dir, f"{split}.pkl")
#         else:
#             raise ValueError("`split` is expected in ['base', 'test', 'val']")
#
#         self.meta = self._gen_meta(pkl_path)
#         self.indices = sorted(self.meta.keys())
#
#     def __len__(self):
#         return len(self.meta)
#
#     def __getitem__(self, item):
#         """ Use this function to retrieve samples by their filename.
#         :param item: list, created by customised sampler
#         :return: (waveforms, label_ids), (list, list)
#         """
#         batch, query_labels = item
#         wavs = []
#         for cluster in batch:
#             wavs.append(self.load_wav(cluster))
#         return wavs, query_labels
#
#     @staticmethod
#     def _gen_meta(pkl_path):
#         """ create a dict containing meta info, format = {`path/to/sound`: `labels`}."""
#         with open(pkl_path, 'rb') as f:
#             data = pickle.load(f)
#
#         meta = dict()
#         for idx, spath in enumerate(data['spath']):
#             meta[spath] = data['labels'][idx]
#         return meta
#
#     @staticmethod
#     def load_wav(wav_path, sr=44100):
#         """ Load waveform from file path(s).
#             :param: wav_path: str or list, path/to/wav file
#             :param: sr: sample rate
#             :returns: array-like or list of array-like
#         """
#         if isinstance(wav_path, list):
#             assert isinstance(wav_path[0], str)
#             wavs = []
#             for wpath in wav_path:
#                 wav, _ = librosa.load(wpath, sr=sr)
#                 wavs.append(wav)
#             return wavs
#         elif isinstance(wav_path, str):
#             wav, _ = librosa.load(wav_path, sr=sr)
#             return wav
#         else:
#             raise TypeError("cannot accept type except str or list")


class FSD50K(NaiveDataset):
    """ FSD50K dataset.
        While other dataset generate wav on the fly, FSD50K generate the wavs and store them in a hdf5 file
        in the first time, as FSD50 need to cut the segments in to fix length and itself is a large-scale dataset.
    """
    def __init__(self,
                 audio_dir: Union[str, Tuple[str, ...], None] = None,
                 csv_path: Union[str, Tuple[str, ...], None] = None,
                 vocabulary_path: Optional[str] = None,
                 clips_dir: Optional[str] = None,
                 mode: Optional[str] = None,
                 num_class: int = 200,
                 sample_rate: int = 22050,
                 overwrite_clips: bool = False
                 ) -> None:
        self.sample_rate = sample_rate
        self.mids2indices, self.mids2labels, self.num_class = FSD50K._make_vocabulary(vocabulary_path)

        if not clips_dir:
            dataset_dir, _ = os.path.split(audio_dir) if isinstance(audio_dir, str) else os.path.split(audio_dir[0])
            clips_dir = os.path.join(dataset_dir, 'clips')

        if not os.path.exists(clips_dir) or overwrite_clips:
            if mode == 'fewshot':
                self.indices = list()
                set = {'clip_names': [], 'clip_wavs': [], 'clip_mids': []}
                for idx, path in enumerate(csv_path):
                    s, i = self.initialisation(audio_dir=audio_dir[idx], csv_path=path, mode=mode, sr=self.sample_rate)
                    # Update each value of set iteratively
                    for k in set.keys():
                        set[k].extend(s[k])
                    self.indices.extend(i)
            else:
                set, self.indices = self.initialisation(audio_dir=audio_dir, csv_path=csv_path, mode=mode, sr=self.sample_rate)
            # Create multi-hot labels
            ids = FSD50K.mid2indices(set['clip_mids'], self.mids2indices)
            set['clip_multihots'] = self.multi_hot(ids, self.num_class).astype(np.bool_) # using bool for less space
            self.clips_path = self.save(clips_dir, set, self.sample_rate, mode)
            # Collect meta info {`clip_name`: `clip_multihots`} note that `clip_name` do not contain '.wav'
            self.meta = dict()
            for i, cname in enumerate(set['clip_names']):
                self.meta[cname] = set['clip_multihots'][i]
            del set  # release the space of `set`
        else:
            self.clips_path = os.path.join(clips_dir, 'wav')
            self.meta, self.indices = self.load_meta(clips_dir, mode)

        self.inverse_indices = dict()
        for idx, cname in enumerate(self.indices):
            self.inverse_indices[cname] = idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: Union[tuple, str]) -> Tuple[Tensor, Tensor]:
        if isinstance(item, tuple):
            cname, multihot = item
            return self.load_wav(cname, self.sample_rate), multihot
        else:
            return self.load_wav(item, self.sample_rate), torch.from_numpy(self.meta[item].astype(np.float32))

    def initialisation(self,
                       audio_dir: Optional[str] = None,
                       csv_path: Optional[str] = None,
                       mode: Optional[str] = None,
                       sr: int = 22050) -> Tuple[dict, list]:
        """ Initialise the dataset by creating fixed-length clips and store them into hdf5 file."""
        df = pd.read_csv(csv_path, header=0)
        if 'split' in df.keys() and (mode == 'train' or mode == 'val'):
            df = df[df['split'] == mode]
        # Collect segment-wise metadata
        seg_names = list()
        seg_wavs = list()
        seg_mids = list()
        for idx in tqdm.trange(len(df)):
            fname = str(df['fname'].iloc[idx])
            wav, ori_sr = torchaudio.load(os.path.join(audio_dir, fname+'.wav'), normalize=True)
            seg_names.append(fname)
            seg_wavs.append(wav)
            seg_mids.append(df['mids'].iloc[idx])
        if not ori_sr == sr:
            resample_fn = torchaudio.transforms.Resample(ori_sr, sr)
            wav_list = list()
            for wav in seg_wavs:
                wav_list.append(resample_fn(wav))
            seg_wavs = wav_list
        # Convert to clip-wise metadata and align them to same length
        clip_duration = int(sr * 1)
        clip_names = list()
        clip_wavs = list()
        clip_mids = list()
        for idx, seg in tqdm.tqdm(enumerate(seg_wavs), total=len(seg_wavs)):
            wavs = fix_len(seg, clip_duration, clip_duration * 0.5)  # overlap = 0.5 * window
            clip_wavs.extend(wavs) # encode the waveform for the future storage
            clip_names.extend([(seg_names[idx] + f"_{i:03d}") for i, _ in
                               enumerate(wavs)])  # encode clip name to store this value in hdf5 file
            clip_mids.extend([seg_mids[idx] for _ in wavs])  # false strong labelling
        assert len(clip_names) == len(clip_wavs) == len(clip_mids)
        set = {'clip_names': clip_names, 'clip_wavs': clip_wavs, 'clip_mids': clip_mids}
        return set, clip_names

    def save(self,
             storage_dir: str,
             set: dict,
             sr: int,
             mode: Optional[str]) -> str:
        print(f"Save clips dataset to folder {storage_dir}")
        clips_dir = os.path.join(storage_dir, 'wav')
        meta_dir = os.path.join(storage_dir, 'meta')
        make_folder(clips_dir)
        make_folder(meta_dir)

        stt = time.time()
        # Save clips in wav files.
        for idx, cname in enumerate(set['clip_names']):
            torchaudio.save(
                filepath=os.path.join(clips_dir, f'{cname}.wav'),
                src=set['clip_wavs'][idx],
                sample_rate=sr,
                encoding='PCM_S'
            )
        # Save the meta info into cpickle
        if mode == 'fewshot':
            meta_path = os.path.join(meta_dir, f'{mode}_meta.pickle')
        elif mode == 'train' or mode == 'val':
            meta_path = os.path.join(meta_dir, f'dev_{mode}_meta.pickle')
        else:
            meta_path = os.path.join(meta_dir, 'eval_meta.pickle')
        with open(meta_path, 'wb') as cf:
            pickle.dump({'clip_names': set['clip_names'], 'clip_multihots': set['clip_multihots']}, cf)
        end = time.time()
        print(f"Clips dataset has been saved, costing {(end - stt)} seconds")
        return clips_dir

    def load_meta(self, dir: str, mode: Optional[str]) -> Tuple[dict, list]:
        """ Load meta info from cpickle file.
            returns meta: dict = {`clip_name`: `clip_onehots`}
                    indices: list = [`cname`]
        """
        print(f"Load metadata from {dir}")
        stt = time.time()
        if mode == 'fewshot':
            meta_path = os.path.join(dir, 'meta', f'{mode}_meta.pickle')
        elif mode == 'train' or mode == 'val':
            meta_path = os.path.join(dir, 'meta', f'dev_{mode}_meta.pickle')
        else:
            meta_path = os.path.join(dir, 'meta', 'eval_meta.pickle')
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        meta = dict()
        for idx, cname in enumerate(data['clip_names']):
            meta[cname] = data['clip_multihots'][idx]
        end = time.time()
        print(f"Metadata has been uploaded, costing {end - stt} seconds")
        return meta, data['clip_names']


    @staticmethod
    def multi_hot(indices, num_class=None):
        """ Convert label idx to one-hot label. e.g.,
        :params indices: list = [[label1, label2, ...], ......]
        :params num_class:
        :return: multi-hot, np.array
        """
        if not num_class:
            max_array = list()
            for ids in indices:
                max_array.append(np.amax(ids))
            num_class = np.amax(max_array) + 1
        # Create multi-hot labels
        audio_num = len(indices)
        multihot = np.zeros(shape=(audio_num, num_class), dtype=np.bool)
        for i, ids in enumerate(indices):
            for id in ids:
                multihot[i][id] = 1
        return multihot

    @classmethod
    def mid2indices(cls, mids, mids2indices):
        """ Convert label mids to indices.
        :param mids: machine id for each categories, see more in audioset ontology
        :param mids2indices: dict = {'mids': 'indices'}
        :return indices: list, e.g., [[idx_0, idx_1, ...], ...]
        """
        indices = list()
        for str in mids:
            tmp = list()
            for m in str.split(','):
                tmp.append(mids2indices[m])
            indices.append(tmp)
        return indices


    @classmethod
    def _make_vocabulary(cls, vocabulary_path):
        """
        to match the mids with indices and their labels, separately
        metadata in FSD50K vocabulary is ['indices', 'labels', 'mids']
        :param vocabulary_path: str, /path/to/your/dataset/ground_truth/vocabulary.csv
        :return mids_to_indices: dict, e.g., {mid1: index1, mid2: index2, ...}
                mids_to_labels: dict, e.g., {mid1: label1, mid2: label2, ...}
                num_class
        """
        voc = pd.read_csv(vocabulary_path, header=None, names=['indices', 'labels', 'mids'])
        mids2indices = {}
        mids2labels = {}
        for idx in voc['indices'][:]:
            mids2indices[voc['mids'].iloc[idx]] = idx
            mids2labels[voc['mids'].iloc[idx]] = voc['labels'].iloc[idx]
        return mids2indices, mids2labels, len(mids2labels)

    def load_wav(self, wav_path: Union[list, str], sr: int, mono: bool=True) -> Union[List[Tensor], Tensor]:
        """ Load waveform from file path(s).
            :param: wav_path: str or list, path/to/wav file
            :param: sr: sample rate
            :returns: array-like or list of array-like
        """
        if isinstance(wav_path, list):
            waveforms = list()
            for wpath in wav_path:
                wav, ori_sr = torchaudio.load(os.path.join(self.clips_path, f"{wpath}.wav"), normalize=True)
                wav = wav.squeeze() if (mono == True and ori_sr == sr) else wav
                waveforms.append(wav)
            # Resample the waveform if sample rate are not matched
            if not ori_sr == sr:
                tmp = list()
                resample_fn = torchaudio.transforms.Resample(ori_sr, sr)
                for wav in waveforms:
                    w = resample_fn(wav)
                    w = w.squeeze() if mono == True else w
                    tmp.append(w)
                waveforms = tmp
            return waveforms
        elif isinstance(wav_path, str):
            wav, ori_sr = torchaudio.load(os.path.join(self.clips_path, f"{wav_path}.wav"), normalize=True)
            if not ori_sr == sr:
                wav = torchaudio.functional.resample(wav, orig_freq=ori_sr, new_freq=sr)
            wav = wav.squeeze() if mono == True else wav
            return wav
        else:
            raise TypeError(f"Cannot load wav(s) in the type of {type(wav_path)} other than `list` and `str`")


class NaiveSampler(object):
    """" An abstract dataset class"""
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class SimpleSampler(NaiveSampler):
    """ Sampler for classical learning."""
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, fix_class=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.fix_class = fix_class

    def __len__(self):
        if self.fix_class:
            slt_ratio = len(
                self.fix_class) / self.dataset.num_class  # calculate the ratio of samples in fixed_class to all
        else:
            slt_ratio = 1

        if self.drop_last:
            return int(len(self.dataset) * slt_ratio // self.batch_size)
        else:
            return int((len(self.dataset) + self.batch_size - 1) * slt_ratio // self.batch_size)

    def __iter__(self):
        if self.fix_class:
            # Sample an instance if its label belongs to the `fix_class`
            indices = list()
            for key, value in self.dataset.meta.items():
                if value in self.fix_class:
                    indices.append(key)
        else:
            indices = self.dataset.indices

        if self.shuffle:
            indices = np.random.permutation(indices)

        batch = []
        for fname in indices:
            batch.append(fname)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


# TODO： batch_size
class ESC50FewShotSampler(NaiveSampler):
    """ Sampler for few shot learning.
        Args:
            dataset: data source, e.g. instance of torch.data.Dataset, data[item] = {filename: labels}
            num_nvl_cls: number of novel classes in an episode (i.e., 'n' ways)
            num_sample_per_cls: number of support samples in each novel class (i.e., 'k' shot)
            num_queries_per_cls: number of queries for each novel class
            num_task: total number of tasks (or episodes) in one epoch
            batch_size: number of tasks (or episodes) in one batch
            fix_split: class indices from which samples would be extracted
            require_pid: return pid along with its id if true. Default is false.
    """
    def __init__(
            self,
            dataset: torch.nn.Module,
            num_nvl_cls: int,
            num_sample_per_cls: int,
            num_queries_per_cls: int,
            num_task: int,
            batch_size: int,
            fix_split: list = None,
            require_pid: bool = False,
            **kwargs
    ) -> None:
        if (batch_size != 1) and require_pid:
            raise ValueError("`batch_size` is expected to be 1 when `require_pid` is True")

        self.dataset = dataset
        self.num_nvl_cls = num_nvl_cls
        self.num_sample_per_cls = num_sample_per_cls
        self.num_queries_per_cls = num_queries_per_cls
        self.num_task = num_task
        self.batch_size = batch_size
        self.fix_split = None
        self.require_pid = require_pid

        if fix_split != None and len(fix_split) > num_nvl_cls:
            self.fix_split = fix_split
        else:
            raise ValueError(f"Make sure `fix_split` contains enough novel classes.")

    def __len__(self):
        """ we assume user can set both params properly so that there won't be reminder at the end."""
        return self.num_task // self.batch_size

    def __iter__(self):
        num_batch = self.num_task // self.batch_size
        for _ in range(num_batch):
            batch = []
            for task in range(self.batch_size):
                if not self.fix_split == None:
                    cls_set = self.fix_split
                else:
                    # Randomly select `num_nvl_cls` novel classes
                    cls_set = list(set(list(self.dataset.meta.values())))
                nvl_cls_ids = np.random.choice(cls_set, size=self.num_nvl_cls, replace=False)
                # Get a subset of metadata containing novel classes only
                subset = {'fname': [], 'label': []}
                for fname, cid in self.dataset.meta.items():
                    if cid in nvl_cls_ids:
                        # `label` = (cid, pid) if `require_pid` is true
                        subset['fname'].append(fname)
                        subset['label'].append(cid)

                subset['fname'] = np.stack(subset['fname'])
                subset['label'] = np.stack(subset['label'])

                # Get ids of support samples
                support_samples = dict()
                for n in nvl_cls_ids:
                    # randomly select 'num_sample_per_cls' support samples from one novel class
                    slt_cls = subset['fname'][subset['label'] == n]
                    samples = np.random.choice(slt_cls, size=self.num_sample_per_cls, replace=False)
                    batch.extend([str(x) for x in list(samples)])

                    support_samples[n] = samples

                # Get ids of query samples
                for n in nvl_cls_ids:
                    # select 'num_queries_per_cls' queries from one novel class
                    # there must be no overlaps (samples) between queries and support samples
                    slt_cls = subset['fname'][subset['label'] == n]
                    samples = np.random.choice(slt_cls[np.isin(slt_cls, support_samples[n], invert=True)],
                                               size=self.num_queries_per_cls, replace=False)
                    batch.extend([str(x) for x in list(samples)])  # match the input format in the class ESC50

                # Create task-specific labels
                support_cids = create_task_label(self.num_nvl_cls, self.num_sample_per_cls)
                query_cids = create_task_label(self.num_nvl_cls, self.num_queries_per_cls)
                batch_y = torch.cat([support_cids, query_cids], dim=0)

                if self.require_pid:
                    _pid, query_pids = list(), list()
                    for n in nvl_cls_ids:
                        pid = self.dataset.pid[n]
                        _pid.append(torch.tensor([pid] * self.num_sample_per_cls))
                        query_pids.append(torch.tensor([pid] * self.num_queries_per_cls))

                    _pid.extend(query_pids)
                    _pid = torch.cat(_pid, dim=0)

                    # Force the label pid from [0, n)
                    _, batch_pid = torch.unique(_pid, sorted=False, return_inverse=True)

                    batch_y = list(zip(batch_y, batch_pid))

            yield list(zip(batch, batch_y))


class One2RestSampler(NaiveSampler):
    """ Sampling on one- v.s. -rest episode for few shot learning.
        See more in "Multi-label Few-shot Learning for Sound Event Recognition, Cheng et al."."""
    def __init__(
            self,
            dataset: torch.nn.Module,
            n_nvl_cls: int,
            n_supports_per_cls: int,
            n_task: int = -1,
            fix_split: Optional[Tuple[int, ...]] = None,
            blacklist: Optional[Tuple[int, ...]] = None,
            is_shuffle: bool = True
    ) -> None:
        self.dataset = dataset
        self.n_nvl_cls = n_nvl_cls
        self.n_supports_per_cls = n_supports_per_cls
        self.is_shuffle = is_shuffle

        # Prepare data points and a label set according to `fix_split`
        if fix_split == 0:
            self.labelset = set(list(range(self.dataset.num_class)))
            self.indices = self.dataset.indices
        else:
            if len(fix_split) < self.n_nvl_cls:
                raise ValueError(f"Make sure `fix_split` exists and contains enough novel classes.")

            if blacklist != 0:
                block_set = set(blacklist)

            self.labelset = set(fix_split)
            covered_set = block_set | self.labelset  # pick a sample only if its labels is subset of `covered_set`
            self.indices = list()
            for instance in self.dataset.indices:
                _labels = set(self.dataset.meta[instance].nonzero()[0].tolist())
                if _labels.issubset(covered_set) and len(_labels & self.labelset) > 0:
                    self.indices.append(instance)
        # Set number of tasks in one epoch
        self.n_task = len(self.indices) if n_task == -1 else n_task

    def __len__(self):
        return self.n_task

    def __iter__(self) -> Tuple[Tuple[str, Tensor], ...]:
        """ Yield batches and ground_truth corresponding to the query containing l unique labels.
            :return: batch: list = [cluster_0, cluster_1, ..., cluster_l] where cluster is a list = [support_0, ..., query]
            :return: query_labels: list of one-hot label in the one- v.s. -rest episode
        """
        if self.is_shuffle:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices

        count = 0
        """ Create l supports subsets corresponding to positive labels of the instance."""
        for instance in indices:
            count += 1
            if count == self.n_task:
                break
            # Construct one-hot label set by picking one of the active labels associated to sample
            # and `n_nvl_cls-1` classes from the non-active labels
            active_labels = list(set(self.dataset.meta[instance].nonzero()[0].tolist()) & self.labelset)
            non_active_labels = list(self.labelset - set(active_labels))

            label_subsets = list()
            for label in active_labels:
                lset = np.random.choice(non_active_labels, size=(self.n_nvl_cls - 1), replace=False).tolist()
                lset.append(label)  # [rand_cls_0, rand_cls_1, rand_cls_2, rand_cls_3, query_label]
                label_subsets.append(np.random.permutation(lset).tolist())  # shuffle the label subset

            # Construct subset where each samples belonging to one and only one cls in the label subset
            subsets = list()
            for lset in label_subsets:
                tmp = dict()
                for fpath in indices:
                    multihot = self.dataset.meta[fpath]
                    labels = multihot.nonzero()[0].tolist()
                    # Strict selection: a samples is selected if and only if one of its labels belongs to label subset.
                    # However, this method is not always work depending on the correlation between labels
                    # if len(set(labels) & set(lset)) == 1:
                    if len(set(labels) & set(lset)) >= 1:
                        tmp[fpath] = labels
                subsets.append(tmp)

            # Sampling from data source
            batch, batch_task_id = list(), list()
            for idx, lset in enumerate(label_subsets):
                # Generate task labels for supports in each cluster
                # e.g., [0]*`n_supports_per_cls`, ..., [`n_nvl_cls`-1]*`n_supports_per_cls`
                batch_task_id.extend(
                    create_task_label(num_cls=self.n_nvl_cls, num_sample_per_cls=self.n_supports_per_cls).tolist()
                )
                cluster = list()
                for i, l in enumerate(lset):
                    # Add relative label idx of queries
                    if l in active_labels:
                        batch_task_id.append(i)

                    slt_samples = []
                    for fpath, labels in subsets[idx].items():
                        if l in labels and instance != fpath:
                            slt_samples.append(fpath)
                    supports_l = np.random.choice(slt_samples, size=self.n_supports_per_cls, replace=False).tolist()
                    cluster.extend(supports_l)
                cluster.append(instance)  # add the only one query to the end and return the cluster to batch
                batch.extend(cluster)

            # assert len(batch) == len(batch_task_id)
            batch_onehot = one_hot(torch.tensor(batch_task_id)).to(torch.float32)

            yield list(zip(batch, batch_onehot))


class HierOne2RestSampler(One2RestSampler):
    """ Sampling on one- v.s. -rest episode for multi-label few-shot learning."""
    def __init__(
            self,
            taxonomy_path: str,
            vocabulary_path: str,
            dataset: torch.nn.Module,
            n_nvl_cls: int,
            n_supports_per_cls: int,
            n_task: int = -1,
            fix_split: Optional[Tuple[int, ...]] = None,
            blacklist: Optional[Tuple[int, ...]] = None,
            is_shuffle: bool = True
    ) -> None:
        super().__init__(
            dataset=dataset,
            n_nvl_cls=n_nvl_cls,
            n_supports_per_cls=n_supports_per_cls,
            n_task=n_task,
            fix_split=fix_split,
            blacklist=blacklist,
            is_shuffle=is_shuffle
        )
        container = FSD50K_MetaContainer(as_taxonomy_path=taxonomy_path, fsd_vocabulary_path=vocabulary_path)
        # Convert to {id: hierarchy} from {mid: state_dict}
        container.curr_taxonomy = container.filter_taxonomy(container.curr_taxonomy, container.curr_vocabulary)  # get fsd taxonomy
        self.taxonomy = dict()
        for mid, state_dict in container.curr_taxonomy.items():
            self.taxonomy[self.dataset.mids2indices[mid]] = {
                'mid': mid,
                'hierarchy': state_dict['hierarchy']
            }
        # Maintain labels in `self.labelset` only
        _black_ids = list(set(list(range(self.dataset.num_class))) - self.labelset)
        _blacklist = list()
        for _label in _black_ids:
            _blacklist.append(self.taxonomy[_label]['mid'])
        container.curr_vocabulary = container.remove_class(container.curr_vocabulary, _blacklist)

        # Convert to [ids, ...] from [mids, ...]
        self.tree = list()
        container.curr_tree = container.filter_tree(container.curr_tree, container.curr_vocabulary)
        for mid_list in container.curr_tree:
            _tmp = list()
            for mid in mid_list:
                _tmp.append(self.dataset.mids2indices[mid])
            self.tree.append(_tmp)


    def __iter__(self) -> Tuple[Tuple[str, Tensor], ...]:
        """ Yield batches and batch_y corresponding to the query containing l unique labels.
            e.g., (batch, batch_y) = (batch_1, Tensor:(multihot_1, hierarchy_1)), ..., (batch_l, Tensor: (multihot_l, hierarchy_l))
        """
        if self.is_shuffle:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices

        """ Create l supports subsets corresponding to positive labels of the instance."""
        count = 0  # Jump out of the batch if `count` is longer than `self.n_task`
        for instance in indices:
            count += 1
            if count == self.n_task:
                break

            # Construct one-hot label set by picking one of the active labels associated to sample
            # and `n_nvl_cls-1` classes from the non-active labels
            active_labels = list(set(self.dataset.meta[instance].nonzero()[0].tolist()) & self.labelset)
            levels, non_active_labels = list(), list()
            for label in active_labels:
                assert len(self.taxonomy[label]['hierarchy']) == 1
                _lvl = self.taxonomy[label]['hierarchy'][0]
                levels.append(_lvl)  # record level of active labels
                _candidate_set = set(self.tree[_lvl]) - set(active_labels)
                assert len(_candidate_set) > (self.n_nvl_cls - 1)
                non_active_labels.append(list(_candidate_set))

            label_subsets = list()
            for id, label in enumerate(active_labels):
                lset = np.random.choice(non_active_labels[id], size=(self.n_nvl_cls - 1), replace=False).tolist()
                lset.append(label)  # [rand_cls_0, rand_cls_1, rand_cls_2, rand_cls_3, query_label]
                label_subsets.append(np.random.permutation(lset).tolist())  # shuffle the label subset

            # Construct subset where each samples belonging to one and only one cls in the label subset
            subsets = list()
            for lset in label_subsets:
                tmp = dict()
                for fpath in indices:
                    multihot = self.dataset.meta[fpath]
                    labels = multihot.nonzero()[0].tolist()
                    # Strict selection: a samples is selected if and only if one of its labels belongs to label subset.
                    # However, this method is not always work depending on the correlation between labels
                    # if len(set(labels) & set(lset)) == 1:
                    if len(set(labels) & set(lset)) >= 1:
                        tmp[fpath] = labels
                subsets.append(tmp)

            # Sampling from data source
            batch, batch_task_id = list(), list()
            for idx, lset in enumerate(label_subsets):
                # Generate task labels for supports in each cluster
                # e.g., [0]*`n_supports_per_cls`, ..., [`n_nvl_cls`-1]*`n_supports_per_cls`
                batch_task_id.extend(
                    create_task_label(num_cls=self.n_nvl_cls, num_sample_per_cls=self.n_supports_per_cls).tolist()
                )
                cluster = list()
                for i, l in enumerate(lset):
                    # Add relative label idx of queries
                    if l in active_labels:
                        batch_task_id.append(i)

                    slt_samples = []
                    for fpath, labels in subsets[idx].items():
                        if l in labels and instance != fpath:
                            slt_samples.append(fpath)
                    supports_l = np.random.choice(slt_samples, size=self.n_supports_per_cls, replace=False).tolist()
                    cluster.extend(supports_l)
                cluster.append(instance)  # add the only one query to the end and return the cluster to batch
                batch.extend(cluster)

            # assert len(batch) == len(batch_task_id)
            batch_onehot = one_hot(torch.tensor(batch_task_id)).to(torch.float32)

            # Append hierarchy information to batch_onehot
            task_size = self.n_supports_per_cls * self.n_nvl_cls + 1
            hierarchy = list()
            for _lvl in levels:
                hierarchy.extend([_lvl] * task_size)
            hierarchy = torch.tensor(hierarchy).view(-1, 1)
            batch_y = torch.cat((batch_onehot, hierarchy), dim=1)  # e.g., batch_y = Tensor: (multihot_l, hierarchy_l)

            yield list(zip(batch, batch_y))


class TaskDependentSampler(One2RestSampler):
    def __init__(
            self,
            taxonomy_path: str,
            vocabulary_path: str,
            label_weights_path: str,
            dataset: torch.nn.Module,
            n_nvl_cls: int,
            n_supports_per_cls: int,
            n_task: int = -1,
            fix_split: Optional[Tuple[int, ...]] = None,
            blacklist: Optional[Tuple[int, ...]] = None,
            is_shuffle: bool = True,
            is_embed_label: bool = False,
            beta: float = 30.0,
            is_aligned: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            n_nvl_cls=n_nvl_cls,
            n_supports_per_cls=n_supports_per_cls,
            n_task=n_task,
            fix_split=fix_split,
            blacklist=blacklist,
            is_shuffle=is_shuffle
        )

        self.is_embed_label = is_embed_label
        if self.is_embed_label:
            self.embed_layer = label_embedding(
                weight_path=label_weights_path,
                n_classes=self.n_nvl_cls,
                beta=beta
            )

        self.is_aligned = is_aligned
        if self.is_aligned:
            with open(label_weights_path, 'rb') as f:
                w = pkl.load(f)
            self.weight_matrix = torch.from_numpy(w)
            self.taxonomy = AudioSetTaxonomy(taxonomy_path, vocabulary_path)
            self.container = FSD50K_MetaContainer(as_taxonomy_path=taxonomy_path, fsd_vocabulary_path=vocabulary_path)

    def __iter__(self) -> Tuple[Tuple[str, Tensor], ...]:
        """ Yield batches and ground_truth corresponding to the query containing l unique labels.
            :return: batch: list = [cluster_0, cluster_1, ..., cluster_l] where cluster is a list = [support_0, ..., query]
            :return: query_labels: list of one-hot label in the one- v.s. -rest episode
        """
        if self.is_shuffle:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices

        count = 0
        """ Create l supports subsets corresponding to positive labels of the instance."""
        for instance in indices:
            count += 1
            if count == self.n_task:
                break
            # Construct one-hot label set by picking one of the active labels associated to sample
            # and `n_nvl_cls-1` classes from the non-active labels
            active_labels = list(set(self.dataset.meta[instance].nonzero()[0].tolist()) & self.labelset)
            non_active_labels = list(self.labelset - set(active_labels))

            # Create a vector to connect a kid task (id) with its parent task (id)
            if self.is_aligned:
                task_size = self.n_supports_per_cls * self.n_nvl_cls + 1
                kp_relation = np.asarray([-1] * len(active_labels))  # initialise parent-kid relation vector as -1
                for idx, _id in enumerate(active_labels):
                    # Find parent ids of the current label in `active_labels`
                    _parent_ids = list()
                    _parent_mids = self.taxonomy.get_parent_mid(mid=self.container.index2mid[_id])
                    for _p_mid in _parent_mids:
                        try:
                            _parent_ids.append(self.container.mid2index(_p_mid))
                        except KeyError:
                            pass
                    # Mark kid ids using the index of their parent ids
                    _is_parent = np.isin(active_labels, _parent_ids)  # restrict parent id to active labels
                    if np.count_nonzero(_is_parent) > 0:
                        assert np.count_nonzero(_is_parent) == 1  # ensure there is no "one-kid-to-multi-parent" case
                        kp_relation[idx] = np.nonzero(_is_parent)[0][0]  # e.g., [-1, parent_id, -1]

            label_subsets = list()
            for idx, label in enumerate(active_labels):
                if self.is_aligned:
                    if kp_relation[idx] != -1:
                        lset = [label]  # contain its own label only if this is a parent category
                    else:
                        lset = np.random.choice(non_active_labels, size=(self.n_nvl_cls - 1), replace=False).tolist()
                        lset.append(label)  # [rand_cls_0, rand_cls_1, rand_cls_2, rand_cls_3, query_label]

                else:
                    lset = np.random.choice(non_active_labels, size=(self.n_nvl_cls - 1), replace=False).tolist()
                    lset.append(label)

                label_subsets.append(np.random.permutation(lset).tolist())  # shuffle the label subset

            # Construct subset where each samples belonging to one and only one cls in the label subset
            subsets = list()
            for idx, lset in enumerate(label_subsets):
                tmp = dict()
                for fpath in indices:
                    multihot = self.dataset.meta[fpath]
                    labels = multihot.nonzero()[0].tolist()
                    if len(set(labels) & set(lset)) >= 1:
                        tmp[fpath] = labels
                subsets.append(tmp)

            # Sampling from data source
            batch = list()
            for idx, lset in enumerate(label_subsets):
                # Sample `n_supports_per_cls` examples as per `lset`
                cluster = list()
                for i, l in enumerate(lset):
                    # Select examples from data subsets
                    slt_samples = []
                    for fpath, labels in subsets[idx].items():
                        if l in labels and instance != fpath:
                            slt_samples.append(fpath)
                    supports_l = np.random.choice(slt_samples, size=self.n_supports_per_cls, replace=False).tolist()
                    cluster.extend(supports_l)
                cluster.append(instance)  # append the query following supports in one task
                batch.append(cluster)  # batch = [[supports_0, query], ..., [supports_n, query]]

            # Align kid tasks with their parent task
            if self.is_aligned:
                for kid_idx, parent_idx in enumerate(kp_relation):
                    if parent_idx != -1:
                        # Replace the parent activate class in `label_subsets` with the kid one
                        _tmp_labels = list()
                        for _idx, _lbl in enumerate(label_subsets[parent_idx]):
                            if _lbl not in active_labels:
                                _tmp_labels.append(_lbl)
                            else:
                                _tmp_labels.append(label_subsets[kid_idx][0])
                                _query_pos_idx = _idx

                        label_subsets[kid_idx] = _tmp_labels

                        # Replace supports of activate class in the parent task with the proper one in the kid task
                        _task_cluster = batch[parent_idx]
                        _task_cluster[
                            self.n_supports_per_cls*_query_pos_idx: self.n_supports_per_cls*(_query_pos_idx+1)
                            ] = batch[kid_idx][:-1]
                        batch[kid_idx] = _task_cluster

            # Stack all elements in `batch`
            _tmp = list()
            for _element in batch:
                _tmp.extend(_element)
            batch = _tmp

            # Generate task labels for supports in each cluster
            batch_onehot = list()
            for idx, lset in enumerate(label_subsets):
                # e.g., [0]*`n_supports_per_cls`, ..., [`n_nvl_cls`-1]*`n_supports_per_cls`
                _batch_task_ids = create_task_label(num_cls=self.n_nvl_cls, num_sample_per_cls=self.n_supports_per_cls)
                batch_onehot.append(one_hot(_batch_task_ids).to(torch.float32))
                # Add relative label idx of queries
                for i, l in enumerate(lset):
                    if l in active_labels:
                        _onehot = one_hot(torch.tensor([i]), num_classes=self.n_nvl_cls).to(torch.float32)
                        if self.is_embed_label:
                            _onehot = self.embed_layer(slt_classes=lset, y=_onehot)
                        batch_onehot.append(_onehot)
            # print(batch_onehot)
            batch_onehot = torch.cat(batch_onehot, dim=0)

            if self.is_aligned:
                # Curate sur-label with parent-kid relation and concatenate it with onehot
                batch_surlabel = [torch.tensor([i] * task_size) for i in kp_relation]
                batch_surlabel = torch.cat(batch_surlabel, dim=0).unsqueeze(dim=1)

                batch_onehot = torch.cat([batch_onehot, batch_surlabel], dim=1)

            yield list(zip(batch, batch_onehot))

#
# class TaskDependentSampler(One2RestSampler):
#     def __init__(
#             self,
#             taxonomy_path: str,
#             vocabulary_path: str,
#             label_weights_path: str,
#             dataset: torch.nn.Module,
#             n_nvl_cls: int,
#             n_supports_per_cls: int,
#             n_task: int = -1,
#             fix_split: Optional[Tuple[int, ...]] = None,
#             blacklist: Optional[Tuple[int, ...]] = None,
#             is_shuffle: bool = True,
#             is_embed_label: bool = False,
#             beta: float = 30.0,
#             is_aligned: bool = False,
#     ) -> None:
#         super().__init__(
#             dataset=dataset,
#             n_nvl_cls=n_nvl_cls,
#             n_supports_per_cls=n_supports_per_cls,
#             n_task=n_task,
#             fix_split=fix_split,
#             blacklist=blacklist,
#             is_shuffle=is_shuffle
#         )
#
#         self.is_embed_label = is_embed_label
#         if self.is_embed_label:
#             self.embed_layer = label_embedding(
#                 weight_path=label_weights_path,
#                 n_classes=self.n_nvl_cls,
#                 beta=beta
#             )
#
#         self.is_aligned = is_aligned
#         if self.is_aligned:
#             with open(label_weights_path, 'rb') as f:
#                 w = pkl.load(f)
#             self.weight_matrix = torch.from_numpy(w)
#             self.taxonomy = AudioSetTaxonomy(taxonomy_path, vocabulary_path)
#             self.container = FSD50K_MetaContainer(as_taxonomy_path=taxonomy_path, fsd_vocabulary_path=vocabulary_path)
#
#     def __iter__(self) -> Tuple[Tuple[str, Tensor], ...]:
#         """ Yield batches and ground_truth corresponding to the query containing l unique labels.
#             :return: batch: list = [cluster_0, cluster_1, ..., cluster_l] where cluster is a list = [support_0, ..., query]
#             :return: query_labels: list of one-hot label in the one- v.s. -rest episode
#         """
#         if self.is_shuffle:
#             indices = np.random.permutation(self.indices)
#         else:
#             indices = self.indices
#
#         count = 0
#         """ Create l supports subsets corresponding to positive labels of the instance."""
#         for instance in indices:
#             count += 1
#             if count == self.n_task:
#                 break
#             # Construct one-hot label set by picking one of the active labels associated to sample
#             # and `n_nvl_cls-1` classes from the non-active labels
#             active_labels = list(set(self.dataset.meta[instance].nonzero()[0].tolist()) & self.labelset)
#             non_active_labels = list(self.labelset - set(active_labels))
#
#             # Align a kid task with its parent task
#             if self.is_aligned:
#                 task_size = self.n_supports_per_cls * self.n_nvl_cls + 1
#                 pk_relation = np.asarray([-1] * len(active_labels))  # initialise parent-kid relation vector as -1
#                 for idx, _id in enumerate(active_labels):
#                     _parent_mids = self.taxonomy.get_parent_mid(mid=self.container.index2mid[_id])
#                     _parent_ids = list()
#                     for _p_mid in _parent_mids:
#                         try:
#                             _parent_ids.append(self.container.mid2index(_p_mid))
#                         except KeyError:
#                             pass
#
#                     # Mark the parent ids using the index of their kid ids
#                     _is_parent = np.isin(active_labels, _parent_ids)  # restrict parent id to active labels
#                     # # Check if there is multi-kid scenario  # todo
#                     # for _i, _bool in enumerate(_is_parent):
#                     #     if _bool:
#                     #         assert pk_relation[_i] == -1
#
#                     if np.any(_is_parent):
#                         np.place(pk_relation, _is_parent, idx)
#
#             label_subsets = list()
#             for idx, label in enumerate(active_labels):
#                 if self.is_aligned and pk_relation[idx] != -1:
#                     child_id = pk_relation[idx]
#                     if child_id > idx:
#                         # swap the value of child and parent position
#                         pk_relation[pk_relation[idx]] = idx
#                         pk_relation[idx] = -1
#
#                         lset = np.random.choice(non_active_labels, size=(self.n_nvl_cls - 1), replace=False).tolist()
#                         lset.append(label)  # [rand_cls_0, rand_cls_1, rand_cls_2, rand_cls_3, query_label]
#
#                     else:
#                         lset = [label]  # contain its own label only if this is a parent category
#
#                 else:
#                     lset = np.random.choice(non_active_labels, size=(self.n_nvl_cls - 1), replace=False).tolist()
#                     lset.append(label)  # [rand_cls_0, rand_cls_1, rand_cls_2, rand_cls_3, query_label]
#
#                 label_subsets.append(np.random.permutation(lset).tolist())  # shuffle the label subset
#
#             # Construct subset where each samples belonging to one and only one cls in the label subset
#             subsets = list()
#             for idx, lset in enumerate(label_subsets):
#                 tmp = dict()
#                 for fpath in indices:
#                     multihot = self.dataset.meta[fpath]
#                     labels = multihot.nonzero()[0].tolist()
#                     if len(set(labels) & set(lset)) >= 1:
#                         tmp[fpath] = labels
#                 subsets.append(tmp)
#
#             # Sampling from data source
#             batch, batch_onehot = list(), list()
#             for idx, lset in enumerate(label_subsets):
#                 # Generate task labels for supports in each cluster
#                 # e.g., [0]*`n_supports_per_cls`, ..., [`n_nvl_cls`-1]*`n_supports_per_cls`
#                 _batch_task_ids = create_task_label(num_cls=self.n_nvl_cls, num_sample_per_cls=self.n_supports_per_cls)
#                 batch_onehot.append(one_hot(_batch_task_ids).to(torch.float32))
#                 # Add relative label idx of queries
#                 for i, l in enumerate(lset):
#                     if l in active_labels:
#                         _onehot = one_hot(torch.tensor([i]), num_classes=self.n_nvl_cls).to(torch.float32)
#                         if self.is_embed_label:
#                             _onehot = self.embed_layer(slt_classes=lset, y=_onehot)
#                         batch_onehot.append(_onehot)
#
#                 cluster = list()
#                 for i, l in enumerate(lset):
#                     # Select examples from data subsets
#                     slt_samples = []
#                     for fpath, labels in subsets[idx].items():
#                         if l in labels and instance != fpath:
#                             slt_samples.append(fpath)
#                     supports_l = np.random.choice(slt_samples, size=self.n_supports_per_cls, replace=False).tolist()
#
#                     if self.is_aligned and pk_relation[idx] != -1:
#                         related_gt = batch_onehot[pk_relation[idx] * 2 + 1]  # batch_onehot = [support_y0, y0, support_y1, y1]
#                         # print(related_gt.size())
#                         batch_onehot[-1] = related_gt
#                         related_pos_idx = torch.nonzero(related_gt == 1)[0, 0]
#
#                         cluster = batch[pk_relation[idx]*task_size: (pk_relation[idx]+1)*task_size-1]  # copy the related batch
#                         cluster[self.n_supports_per_cls*related_pos_idx: self.n_supports_per_cls*(related_pos_idx+1)] = supports_l
#                     else:
#                         cluster.extend(supports_l)
#                 cluster.append(instance)
#                 batch.extend(cluster)
#
#             # print(batch_onehot)
#             batch_onehot = torch.cat(batch_onehot, dim=0)
#
#             if self.is_aligned:
#                 # Curate sur-label with parent-kid relation and concatenate it with onehot
#                 batch_surlabel = [torch.tensor([i] * task_size) for i in pk_relation]
#                 batch_surlabel = torch.cat(batch_surlabel, dim=0).unsqueeze(dim=1)
#
#                 batch_onehot = torch.cat([batch_onehot, batch_surlabel], dim=1)
#
#             yield list(zip(batch, batch_onehot))


# todo: add batch_size
class MultiLabelFewShotSampler(NaiveSampler):
    """ A plain data sampler for few shot learning.
        Arg:
            dataset (NaiveDataset), returns file name and
                contains attribute *.meta = {`fname`: `multihot`}
                                   *.num_class indicate the number of classes in the dataset
                returns Tuple of `array-like or str` and `multihot` label
            n_nvl_cls (int), number of novel classes in a few shot learning task
            n_support_per_cls (int), number of support samples per novel class
            n_query_per_cls (int), number of query samples per novel class
            n_task (int), number of tasks
            batch_size (int) number of task per batch, default = 1
            fix_split, list of class ids, default = None
            shuffle_class (bool), if shuffling the label set when constituting novel set, default = True,
                keep it unchanged unless `n_nvl_cls` = len(`fix_split`) in evaluation.
    """
    def __init__(
            self,
            dataset: NaiveDataset,
            n_nvl_cls: int,
            n_support_per_cls: int,
            n_query_per_cls: int,
            n_task: int,
            batch_size: int,
            fix_split: Optional[Tuple[int, ...]] = None,
            blacklist: Optional[Tuple[int, ...]] = None,
            shuffle_class: bool = True
    ) -> None:
        self.dataset = dataset
        self.n_nvl_cls = n_nvl_cls
        self.n_support_per_cls = n_support_per_cls
        self.n_query_per_cls = n_query_per_cls
        self.n_task = n_task
        self.batch_size = 1
        self.fix_split = fix_split
        self.shuffle_class = shuffle_class

        # Prepare data points and a label set according to `fix_split`
        if self.fix_split == 0:
            self.indices = self.dataset.indices
        else:
            if len(self.fix_split) < self.n_nvl_cls:
                raise ValueError(f"Make sure `fix_split` exists and contains enough novel classes.")

            block_set = set(blacklist) if blacklist else set()

            labelset = set(self.fix_split)
            covered_set = block_set | labelset  # pick a sample only if its labels is subset of `covered_set`
            self.indices = list()
            for fname, multihot in self.dataset.meta.items():
                _labels = set(multihot.nonzero()[0].tolist())
                if _labels.issubset(covered_set) and len(_labels & labelset) > 0:
                    self.indices.append(fname)

    def __len__(self):
        return self.n_task // self.batch_size

    def __iter__(self) -> Tuple[Tuple[str, np.ndarray], ...]:
        # Create a list of selected labels containing all available labels in the data source
        for _ in range(self.n_task // self.batch_size):
            # batch = list()
            for _ in range(self.batch_size):
                cluster = list()
                if self.shuffle_class:
                    nvl_cls_ids = np.random.choice(self.fix_split, size=self.n_nvl_cls, replace=False)
                else:
                    assert self.n_nvl_cls == len(self.fix_split)
                    nvl_cls_ids = self.fix_split

                subset = {'fname': [], 'multihot': []}
                # ***This is strict selection, we found it hard to use as the implicit correlations between labels.***
                # Get a subset of metadata containing novel classes only,
                # one sample be included only if all its labels belong to the novel split
                # for fname, multihot in self.dataset.meta.items():
                #     if np.all(np.isin(multihot.nonzero(), nvl_cls_ids)):
                #         subset['fname'].append(fname)
                #         subset['multihot'].append(multihot)
                #
                # ***A tolerant selection, take a sample into subset if one of its labels belongs to `nvl_cls_ids`.***
                for instance in self.indices:
                    if np.any(np.isin(self.dataset.meta[instance].nonzero()[0], nvl_cls_ids)):
                        subset['fname'].append(instance)
                        subset['multihot'].append(self.dataset.meta[instance])
                # ****************************************************************************************************
                # Get ids of support samples
                for n in nvl_cls_ids:
                    slt_cls = list()
                    for i, fname in enumerate(subset['fname']):
                        if subset['multihot'][i][n]:
                            slt_cls.append(fname)
                    # Randomly select 'n_support_per_cls' support samples from one novel class
                    samples = np.random.choice(slt_cls, size=self.n_support_per_cls, replace=False).tolist()
                    cluster.extend(samples)
                supports = cluster # record selected supports to ensure it will not appear in the queries
                # Get ids of queries
                for n in nvl_cls_ids:
                    # Select 'n_query_per_cls' queries per novel class,
                    # there must be no overlaps between queries and supports
                    slt_cls = list()
                    for i, fname in enumerate(subset['fname']):
                        if subset['multihot'][i][n] and (fname not in supports):
                            slt_cls.append(fname)
                    samples = np.random.choice(slt_cls, size=self.n_query_per_cls, replace=False).tolist()
                    cluster.extend(samples)
                # Collate samples with their task label
                task_multihots = torch.tensor(self.task_multihot(cluster, nvl_cls_ids), dtype=torch.float32)
                outputs = list(zip(cluster, task_multihots))
                batch = outputs
                # batch.append(cluster)
            yield batch

    def task_multihot(self, filelist: Tuple[str, ...], slt_cls: Tuple[int, ...]) -> np.ndarray:
        """ Read multihot from meta and convert multihots to ones with limited class number in an independent task."""
        n_file, n_class = len(filelist), len(slt_cls)
        task_multihot = np.zeros((n_file, n_class), dtype=np.bool_)
        for f_id, file in enumerate(filelist):
            multihot = self.dataset.meta[file]
            for i, cls_id in enumerate(slt_cls):
                task_multihot[f_id, i] = multihot[cls_id]
        return task_multihot



if __name__ == '__main__':
    """ This is a test module."""
    # Select dataset and/or sampler for test
    dataset_name = 'fsd50k'
    sampler_name = 'simple'
    # Set porams required by the selected class
    dataset_dir = '/data/EECS-MachineListeningLab/datasets/FSD50K'
    hdf5_path = os.path.join(dataset_dir, 'hdf5', 'dev_val.h5')
    audio_dir = os.path.join(dataset_dir, 'dev_audio')
    csv_path = os.path.join(dataset_dir, 'ground_truth', 'dev.csv')
    vocabulary_path = os.path.join(dataset_dir, 'ground_truth', 'vocabulary.csv')
    mode = 'val'

    logger = set_logger('debug', os.path.join(dataset_name, sampler_name))
    logger.info(f"Now test {dataset_name} dataset using {sampler_name} sampler")

    testDataset = dataset_cls(dataset_name)
    testSampler = sampler_cls(sampler_name)

    test_dataset = testDataset(hdf5_path, audio_dir=audio_dir, csv_path=csv_path, vocabulary_path=vocabulary_path,
                               mode=mode)
    test_sampler = testSampler(test_dataset, batch_size=32, shuffle=True, drop_last=False, fix_class=None)
    import torch

    data_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler)
    for _ in range(1):
        x, y = next(iter(data_loader))
        print(y)
        print(len(y))
        print(x[0].size())

    # with open('fewshotESC50.yaml', 'r') as f:
    #     cfgs = yaml.safe_load(f)
    # # print("Test dataset and sampler by initialization...")
    # # # dirs and paths
    # # dataset_dir = cfgs.DATASET_DIR
    # # wav_dir = os.path.join(dataset_dir, 'audio')
    # #
    # # # dataset and hyper-params
    # # csv_path = os.path.join(dataset_dir, 'meta', 'esc50.csv')
    # # fs_dataset = ESC50(wav_dir, csv_path, [str(x) for x in range(1, 6)])
    # # fs_sampler = FewShotSampler(fs_dataset, num_nvl_cls=15, num_sample_per_cls=5, num_queries_per_cls=5, num_task=100, batch_size=2)
    # #
    # DATASET_DIR = '/import/c4dm-datasets/FSD-MIX/FSD_MIX_OWN/poly_2'
    # pkl_path = os.path.join(DATASET_DIR, 'FSD_MIX_SED.annotation/ground_truth')
    # fs_dataset = FSD_MIX(pkl_path, 'base', 'test')
    #
    # fs_sampler = MLFewShotSampler(fs_dataset, 5, 5, 5)
    #
    # import torch
    # fs_loader = torch.utils.data.DataLoader(fs_dataset, sampler=fs_sampler)
    #
    # for _ in range(1):
    #     x = next(iter(fs_loader))
    #     # print(x)
    #     # print(len(x))
    #     # print(len(x[0]))