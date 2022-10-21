import os
import csv
import numpy as np
from typing import Callable, List, Tuple
from src.esc_meta import esc50_select_ids
from src.fsd_meta import FSD50K_MetaContainer, fsd50k_select_ids

# Fix random seed
np.random.seed(1)


def parent_split(csv_path: str, prnt_id: int, n_novel_cls: int) -> Tuple[list, list]:
    """ Split classes mainly by their parent classes,
        and randomly append from the rest of label set if max(len(parent_id==nvl_id)) < n_novel_cls."""
    _, _, vocabulary = create_vocabulary(csv_path)
    base_split, novel_split = split_by_parent(vocabulary, nvl_parent_id=prnt_id)
    if len(novel_split) < n_novel_cls:
        base_split, novel_ids = random_split(base_split, int(n_novel_cls - len(novel_split)))
        novel_split.extend(novel_ids)
    elif len(novel_split) > n_novel_cls:
        base_ids, novel_split = random_split(novel_split, int(len(novel_split) - n_novel_cls))
        base_split.extend(base_ids)
    assert (set(base_split) & set(novel_split) == set())
    return base_split, novel_split


def split_by_parent(vocabulary, nvl_parent_id=None):
    """ split classes by parent category according to tree topology.
    :param vocabulary: dict = {'class_name': {'pid', 'id'}}
    :param nvl_parent_id: select a parent id and consider its child classes as novel classes
    :return: tuple = (base, novel)
    """
    base = []
    novel = []
    if nvl_parent_id == None:
        nvl_parent_id = np.random.randint(5) # type of `pid` is str
        print(f"randomly select classes under pid={nvl_parent_id} as novel classes.")
    for key, value in vocabulary.items():
        if value['pid'] == nvl_parent_id:
            # novel.append({key: value['id']})
            novel.append(int(value['id']))
        else:
            base.append(int(value['id']))
    return base, novel


def random_split(class_ids: list, n_novel_cls: int) -> Tuple[list, list]:
    """ Select classes randomly."""
    shuffled_ids = np.random.permutation(class_ids)
    novel_split = shuffled_ids[:n_novel_cls].tolist()
    base_split = shuffled_ids[n_novel_cls:].tolist()
    assert (set(base_split) & set(novel_split) == set())
    return base_split, novel_split


def uniform_split(csv_path: str, n_novel_cls: int) -> Tuple[List[int], List[int]]:
    """ Selected classes from each parent class uniformly to ensure their distribution will be same."""
    _, parent_voc, vocabulary = create_vocabulary(csv_path)
    n_parent_cls = len(parent_voc)
    base_split, novel_split = list(), list()
    for pid in range(n_parent_cls):
        _, novel_ids = split_by_parent(vocabulary, nvl_parent_id=pid)
        novel_ids = np.random.permutation(novel_ids)
        novel_split.extend(novel_ids[:int(n_novel_cls / n_parent_cls)])
        base_split.extend(novel_ids[int(n_novel_cls / n_parent_cls):])
    assert (set(base_split) & set(novel_split) == set())
    return base_split, novel_split


if __name__ == '__main__':
    label_split = 'select'
    csv_path = os.path.join('/import/c4dm-datasets/ESC50', 'meta', 'esc50.csv')
    n_class = 50
    K = 5
    n_novel_cls = 15

    for fold in range(K):
        if label_split == 'random':
            base_split, novel_split = random_split(list(range(n_class)), n_novel_cls=n_novel_cls)
        elif label_split == 'uniform':
            base_split, novel_split = uniform_split(csv_path=csv_path, n_novel_cls=n_novel_cls)
        elif label_split == 'parent':
            base_split, novel_split = parent_split(csv_path=csv_path, prnt_id=(fold + 1), n_novel_cls=n_novel_cls)
        elif label_split == 'select':
            assert len(esc50_select_ids) == K
            novel_split = esc50_select_ids[fold]
            base_split = [x for x in range(n_class) if x not in novel_split]
            assert (set(base_split) & set(novel_split) == set())
        print(f"base split contains {base_split}")
        print(f"novel split contains {novel_split}\n")