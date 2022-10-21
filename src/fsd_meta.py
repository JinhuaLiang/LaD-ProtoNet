import json
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

# Fix random seed
np.random.seed(1)


fsd50k_select_ids = {
    'base': [37, 33, 75, 198, 84, 85, 43, 111, 107, 148, 186, 77, 30, 44, 31, 21, 29, 39, 190, 157, 150, 22, 149, 158,
             131, 91, 116, 15, 7, 50, 14, 82, 28, 59, 126, 57, 112, 67, 181, 90, 166, 88, 20, 125, 104, 144, 96, 127,
             97, 135, 134, 122, 162, 177, 119, 180, 42, 182, 26, 23, 153, 178, 140, 12, 65, 155, 154, 64, 3, 83, 55,
             40, 169, 147, 197, 199, 17, 183, 136, 129, 10, 94, 16, 35, 152, 117, 102, 62, 0, 124, 79, 118, 133, 92, 143,
             146, 52, 168],
    'novel_val': [110, 159, 6, 73, 51, 54, 49, 89, 103, 68, 138, 195, 1, 189, 93, 179, 164, 80, 76, 71, 160, 130,
                  105, 151, 114, 56, 188, 128, 58, 53],
    'novel_eval': [74, 145, 141, 32, 81, 123, 19, 171, 174, 142, 175, 41, 18, 25, 66]
}

fsd50k_blacklist = [100, 196, 109, 167, 191, 165, 70, 13, 69, 99, 156, 108, 115, 36, 163, 4, 120, 5, 86, 113, 161, 45,
                    172, 193, 60, 101, 121, 170, 192, 72, 38, 48, 98, 9, 184, 106, 132, 27, 87, 24, 63, 176, 11, 194,
                    187, 173, 34, 46, 95, 139, 61, 2, 47, 137, 78, 185, 8]


class FSD50K_MetaContainer(object):
    """ Meta Container and all operations devised for FSD50K."""
    def __init__(
        self,
        as_taxonomy_path: str,
        fsd_vocabulary_path: str,
    ) -> None:
        # Set up AudioSet taxonomy & tree
        self.as_taxonomy = init_as_taxonomy(as_taxonomy_path)
        self.as_tree = setup_tree_structure(self.as_taxonomy)
        self.as_taxonomy = measure_height(self.as_taxonomy, self.as_tree)
        # Set up FSD50K vocabulary
        self.fsd_vocabulary = FSD50K_MetaContainer.init_vocabulary(fsd_vocabulary_path)
        self.fsd_tree = FSD50K_MetaContainer.filter_tree(self.as_tree, self.fsd_vocabulary)
        # Define currently working attributes
        self.curr_vocabulary = self.fsd_vocabulary.copy()
        self.curr_tree = self.fsd_tree.copy()
        self.curr_taxonomy = self.as_taxonomy.copy()
        # Generate some frequently-used dictionaries
        self.index2mid, self.index2label = self.create_dictionaries()

    def create_dictionaries(self):
        index2mid, index2label = dict(), dict()
        for _k, _v in self.curr_vocabulary.items():
            index2label[_v['index']] = _v['label']
            index2mid[_v['index']] = _k

        return index2mid, index2label

    @classmethod
    def init_vocabulary(cls, path_to_csv: str) -> dict:
        """ Init the fsd50k dataset."""
        vocabulary = {}
        with open(path_to_csv, 'r') as f1:
            meta = pd.read_csv(f1, names=['index', 'label', 'id'])
            _indices = meta['index'].values.tolist()
            _labels = meta['label'].values.tolist()
            _ids = meta['id'].values.tolist()
            for index, label, id in zip(_indices, _labels, _ids):
                vocabulary[id] = {
                    'index': index,
                    'label': label
                }
        return vocabulary

    @classmethod
    def filter_taxonomy(cls, ori_taxonomy: dict, vocabulary: dict) -> dict:
        """ Filter taxonomy according to the provided vocabulary."""
        mid_set = list(vocabulary.keys())
        output = dict()
        for mid, cat_dict in ori_taxonomy.items():
            if mid in mid_set:
                output[mid] = cat_dict
        assert len(output) == len(vocabulary)
        return output

    @classmethod
    def filter_tree(cls, ori_tree: list, vocabulary: dict) -> list:
        """ Filter tree according to the provided vocabulary."""
        mid_set = set(list(vocabulary.keys()))
        output = [None] * len(ori_tree)  # create an empty list which has the same length as `ori_tree`
        for lvl, l_set in enumerate(ori_tree):
            output[lvl] = list((set(l_set) & mid_set))
        return output

    @classmethod
    def remove_class(cls, ori_vocabulary: dict, black_list: dict) -> dict:
        """ Remove classes from ori_vocabulary as per black_list."""
        for mid in black_list:
            del ori_vocabulary[mid]
        return ori_vocabulary

    def mid2index(self, mid: Union[str, list]) -> Union[str, list]:
        """ Convert mid/list of mids to index/list of indices."""
        def _mid_to_index(mid: str, voc: dict) -> int:
            return int(voc[mid]['index'])

        if isinstance(mid, str):
            return _mid_to_index(mid, self.fsd_vocabulary)
        elif isinstance(mid, list):
            res = list()
            for m in mid:
                res.append(_mid_to_index(m, self.fsd_vocabulary))
            return res
        else:
            raise TypeError(f"Cannot handle mid of {type(mid)}")

    @property
    def multipath_class(self) -> Optional[list]:
        curr_taxonomy = FSD50K_MetaContainer.filter_taxonomy(self.as_taxonomy, self.fsd_vocabulary)
        multipath_mids = list()
        for mid, cat_dict in curr_taxonomy.items():
            if len(cat_dict['hierarchy']) != 1:
                multipath_mids.append(mid)
        return multipath_mids

    def show_tree(self) -> None:
        for lvl, l_set in enumerate(self.curr_tree):
            print(f"{lvl}-th/{len(self.curr_tree)} contains ({len(l_set)}/{len(self.curr_vocabulary)}) classes:"
                  f" {l_set}.\n")

    def trace_ancestor(self, level: int, update_taxonomy: bool = False) -> dict:
        """ Trace ancestor categories under a specific level of AudioSet tree."""
        as_trace = dict()
        for ancestor in self.as_tree[level]:
            _tmp = [[] for _ in range(level+1)]
            _tmp[level].append(ancestor)
            for lvl in range(level, 0, -1):
                # Retrieve a class if it is a child class under a specific ancestor according to AudioSet taxonomy
                for mid in _tmp[lvl]:
                    for c_mid in self.as_taxonomy[mid]['child_mid']:
                        _tmp[lvl-1].append(c_mid)
                        if update_taxonomy:
                            try:
                                self.curr_taxonomy[c_mid]['ancestor_mid'].append(ancestor)
                            except KeyError:
                                pass
                _tmp[lvl-1] = list(set(_tmp[lvl-1]))
            as_trace[ancestor] = _tmp
        # Remove duplicate, identical ancestor in the taxonomy
        if update_taxonomy:
            for cat_dict in self.curr_taxonomy.values():
                cat_dict['ancestor_mid'] = list(set(cat_dict['ancestor_mid']))

        # Filter mids in `as_trace` to ensure all elements belonging to `current_tree`
        res_trace = dict()
        for ancestor, branches in as_trace.items():
            _tmp = [[] for _ in branches]
            for i, branch_i in enumerate(branches):
                intersect = set(branch_i) & set(self.curr_tree[i])
                _tmp[i].extend(list(intersect))
            res_trace[ancestor] = _tmp

        return res_trace

    def uniform_split(self):
        pass


def init_as_taxonomy(path_to_json) -> dict:
    """ Initialise taxonomy with AudioSet taxonomy.
        E.g.
            taxonomy = init_taxonomy(path_to_json) # taxonomy = {
                'mid': {'name': , 'child_mids':, 'restrictions':, 'parent_mids':, 'ancestor_mid':},
                ...
                }
    """
    taxonomy = {}
    with open(path_to_json, 'rb') as f:
        meta = json.load(f)
    for cat in meta:
        _tmp = {}
        _tmp['name'] = cat['name']
        _tmp['child_mid'] = cat['child_ids']
        _tmp['restrictions'] = cat['restrictions']
        _tmp['parent_mid'] = []
        _tmp['ancestor_mid'] = []

        taxonomy[cat['id']] = _tmp

    return taxonomy

def setup_tree_structure(taxonomy: dict) -> Tuple[dict]:
    """ Initiate tree structure by tracing the parent-child relationship within a taxonomy."""
    _lvl = 0
    inverse_tree = dict()
    # Trace parent mids for each mid
    for mid in taxonomy.keys():
        for child_mid in taxonomy[mid]['child_mid']:
            taxonomy[child_mid]['parent_mid'].append(mid)
    # Trace ancestor nodes in taxonomy
    _tmp = list()
    for mid, cat_dict in taxonomy.items():
        if cat_dict['parent_mid'] == []:
            _tmp.append(mid)
    inverse_tree[_lvl] = _tmp

    # Iter to trace the level of each category in the taxonomy,
    # note some of them may belong to multiple levels (multi-path to root)
    while True:
        _lvl += 1
        _tmp = list()
        for mid, cat_dict in taxonomy.items():
            for pmid in cat_dict['parent_mid']:
                if pmid in inverse_tree[_lvl - 1]:
                    _tmp.append(mid)
        if _tmp != []:
            inverse_tree[_lvl] = set(_tmp)
        else:
            break
    # Get the tree structure where the bottom level is denoted as 0
    tree = list()
    height = len(inverse_tree)
    for lvl in range(height):
        tree.append(inverse_tree[height - lvl - 1])

    return tree

def measure_height(taxonomy: dict, tree: list) -> dict:
    """ Measure the height of each class in the tree structure."""
    for mid, cat_dict in taxonomy.items():
        _tmp = list()
        for id, lvl in enumerate(tree):
            if mid in lvl:
                _tmp.append(id)
        cat_dict['hierarchy'] = list(set(_tmp))

    return taxonomy


if __name__ == '__main__':
    container = FSD50K_MetaContainer(
        as_taxonomy_path='./taxonomy/ontology.json',
        fsd_vocabulary_path='./taxonomy/ground_truth/vocabulary.csv')

    # print(f"---Before blacklist---")
    # for i, tree_i in enumerate(container.curr_tree):
    #     print(f"{i}-th/{len(container.curr_tree)} tree contains {len(tree_i)} mids / {len(container.curr_vocabulary)}")

    # Blacklist some classes
    # multipath_mids = container.multipath_class
    # intra_overlaps = ['/m/07pzfmf', '/m/07q6cd_', '/m/07qjznt', '/m/07qs1cx']
    # # intra_overlaps = []
    # black_list = list(set([*container.fsd_tree[1], *container.fsd_tree[4], *container.fsd_tree[5], *multipath_mids, *intra_overlaps]))  # remove overlaps
    # container.curr_vocabulary = container.remove_class(container.curr_vocabulary, black_list)
    # print(len(container.curr_vocabulary))
    # container.curr_tree = container.filter_tree(container.curr_tree, container.curr_vocabulary)
    #
    # print(f"---After blacklist---")
    # for i, tree_i in enumerate(container.curr_tree):
    #     if len(tree_i) != 0:
    #         print(f"{i}-th/{len(container.curr_tree)} tree contains {len(tree_i)} mids")
    #
    # ancestor_map = container.trace_ancestor(level=5, update_taxonomy=True)

    # for key, value in ancestor_map.items():
    #     count = 0
    #     for v in value[:-1]:
    #         count += len(v)
    #     print(f"{container.as_taxonomy[key]['name']} ({key}) contains {count} classes")

    # # find the overlaps between ancestor
    # keys = list(ancestor_map.keys())
    # for key in keys:
    #     rest_map = ancestor_map
    #     value = rest_map.pop(key)
    #     for v in rest_map.values():
    #         for lvl in range(len(container.curr_tree)):
    #             intersect = set(value[lvl]) & set(v[lvl])
    #             if len(intersect) != 0:
    #                 print(intersect)

    # split data
    # n_base_cls = {
    #     '/m/0dgw9r': [8, 16],
    #     '/m/0jbk': [6, 4],
    #     '/m/04rlf': [8, 7],
    #     '/m/059j3w': [1, 3],
    #     '/t/dd00041': [14, 28],
    #     '/t/dd00098': [1, 2]
    # }
    # n_nvlval_cls = {
    #     '/m/0dgw9r': [2, 4],
    #     '/m/0jbk': [2, 1],
    #     '/m/04rlf': [2, 2],
    #     '/m/059j3w': [1, 1],
    #     '/t/dd00041': [3, 11],
    #     '/t/dd00098': [0, 1]
    # }
    # n_nvleval_cls = {
    #     '/m/0dgw9r': [1, 2],
    #     '/m/0jbk': [1, 1],
    #     '/m/04rlf': [1, 1],
    #     '/m/059j3w': [0, 1],
    #     '/t/dd00041': [1, 6],
    #     '/t/dd00098': [0, 0]
    # }
    #
    # for key in n_base_cls.keys():
    #     for lvl in range(2, 4):
    #         print(f"{container.as_taxonomy[key]['name']} ({key}): "
    #               f"{lvl}: {len(ancestor_map[key][lvl])}"
    #               f">>{container.mid2index(ancestor_map[key][lvl])}")

    # base_split = {
    #     '/m/0dgw9r': [],
    #     '/m/0jbk': [],
    #     '/m/04rlf': [],
    #     '/m/059j3w': [],
    #     '/t/dd00041': [],
    #     '/t/dd00098': ['/m/07pzfmf', '/m/07qs1cx']
    # }
    #
    # nvlval_split = {
    #     '/m/0dgw9r': [],
    #     '/m/0jbk': [],
    #     '/m/04rlf': [],
    #     '/m/059j3w': [],
    #     '/t/dd00041': [],
    #     '/t/dd00098': ['/m/07q6cd_']
    # }
    #
    # nvleval_split = {
    #     '/m/0dgw9r': [],
    #     '/m/0jbk': [],
    #     '/m/04rlf': [],
    #     '/m/059j3w': [],
    #     '/t/dd00041': [],
    #     '/t/dd00098': ['/m/07qjznt']
    # }
    # # base_split, nvlval_split, nvleval_split = list(), list(), list()
    # for key in n_base_cls.keys():
    #     for lvl in range(2, 4):
    #         _tmp = ancestor_map[key][lvl]
    #         _tmp = np.random.permutation(_tmp)
    #         base_split[key].append(
    #             container.mid2index(_tmp[:n_base_cls[key][lvl-2]].tolist())
    #         )
    #         nvlval_split[key].append(
    #             container.mid2index(_tmp[n_base_cls[key][lvl-2]: n_base_cls[key][lvl-2]+n_nvlval_cls[key][lvl-2]].tolist())
    #         )
    #         nvleval_split[key].append(
    #             container.mid2index(_tmp[n_base_cls[key][lvl-2]+n_nvlval_cls[key][lvl-2]:].tolist())
    #         )
    #
    # print(f"base_split ({len(base_split)}): {base_split}")
    # print(f"nvlval_split ({len(nvlval_split)}): {nvlval_split}")
    # print(f"nvleval_split ({len(nvleval_split)}): {nvleval_split}")
    # fsd_split = {
    #     'base': base_split,
    #     'novel_val': nvlval_split,
    #     'novel_eval': nvleval_split
    # }
    # print(f"=========FSD_DATA_SPLIT=========\n"
    #       f"{fsd_split}")

    # OUTPUT:
    # {
    #     'base': [
    #         '/m/07rgt08', '/m/0ytgt', '/m/02zsn', '/m/07sr1lc', '/m/07s0dtb', '/m/07r660_', '/m/01h8n0', '/m/05zppz',
    #         '/m/01j3sz', '/m/03qc9zr', '/m/07pbtc8', '/m/025_jnm', '/m/053hz1', '/m/01b_21', '/m/03cczk', '/m/03qtwd',
    #         '/m/07rkbfh', '/m/0l15bq', '/m/02rtxlg', '/m/01hsr_', '/m/015lz1', '/m/03q5_w', '/m/07plz5l', '/m/028ght',
    #         '/m/02yds9', '/m/01dwxx', '/m/07qrkrw', '/m/020bb7', '/m/05tny_', '/m/04s8yn', '/m/025rv6n', '/m/09ld4',
    #         '/m/01yrx', '/m/0bt9lr', '/m/05r5c', '/m/01qbl', '/m/0j45pbj', '/m/026t6', '/m/07gql', '/m/0342h',
    #         '/m/07brj', '/m/0mbct', '/m/0l14_3', '/m/0l14md', '/m/05148p4', '/m/01hgjl', '/m/03qjg', '/m/0mkg',
    #         '/m/03m5k', '/m/07r10fb', '/m/06mb1', '/m/05kq4', '/m/0j6m2', '/m/0btp2', '/m/04_sv', '/m/07r04',
    #         '/m/01m2v', '/m/0c2wf', '/m/0k4j', '/m/01bjv', '/m/06_fw', '/m/07jdr', '/m/01hnzm', '/m/0199g',
    #         '/m/01d380', '/m/02y_763', '/m/07rjzl8', '/m/0fqfqc', '/m/07prgkl', '/m/0dxrf', '/m/0642b4', '/m/01x3z',
    #         '/m/07p7b8y', '/m/07rn7sz', '/m/081rb', '/m/01s0vc', '/m/0dv3j', '/m/01b82r', '/m/02bm9n', '/m/0_ksk',
    #         '/m/03dnzn', '/m/03l9g', '/m/019jd', '/m/07q7njn', '/m/03kmc9', '/m/0fx9l', '/m/07pb8fc', '/m/02dgv',
    #         '/m/07q2z82', '/m/05mxj0q', '/m/0g6b5', '/m/0dv5r', '/m/06d_3', '/m/032s66', '/m/01lsmm', '/m/07q8k13',
    #         '/t/dd00112', '/m/07qcx4z'
    #     ],
    #     'novel_val': [
    #         '/t/dd00003', '/m/0brhx', '/m/09x0r', '/m/02_nn', '/m/0lyf6', '/m/0463cq4', '/m/09b5t', '/m/0ghcn6',
    #         '/m/03vt0', '/m/02hnl', '/m/05r5wn', '/m/085jw', '/m/0fx80y', '/m/034srq', '/m/07swgks',
    #         '/m/07pqc89', '/m/0195fx', '/m/0cmf2', '/m/07cx4', '/t/dd00130', '/m/07rrlb6', '/m/01m4t',
    #         '/m/03v3yw', '/m/0130jx', '/m/02x984l', '/m/023pjk', '/m/02jz0l', '/m/0k5j', '/m/04brg2',
    #         '/m/07plct2'
    #     ],
    #     'novel_eval': [
    #         '/t/dd00004', '/m/07p6fty', '/m/06h7j', '/m/09xqv', '/m/015p6', '/m/013y1f', '/m/01kcd', '/m/0ngt1',
    #         '/m/07qjznl', '/m/0316dw', '/m/01jt3m', '/m/0242l', '/m/07qqyl4', '/m/012f08', '/m/07r5v4s'
    #     ]
    # }

    # validate no overlaps between split
    # base_split = set(base_split)
    # nvlval_split = set(nvlval_split)
    # nvleval_split = set(nvleval_split)
    # assert len(base_split & nvlval_split) == 0
    # assert len(base_split & nvleval_split) == 0
    # assert len(nvlval_split & nvleval_split) == 0
    # print(len([*base_split, *nvlval_split, *nvleval_split]))

    # Black list in fsd
    # labelset = list(container.fsd_vocabulary.keys())
    #
    # selected_set = list()
    # for v in fsd_split.values():
    #     selected_set += v
    #
    # fsd_blacklist = list(set(labelset) - set(selected_set))
    # print(f"=========FSD BLACK LIST=========\n"
    #       f"{fsd_blacklist}")
    # print(len(fsd_blacklist))
    # OUTPUT:
    # fsd50k_blacklist = ['/t/dd00012', '/m/083vt', '/m/0ch8v', '/m/07qcpgn', '/m/07rqsjt', '/m/01p970', '/m/02mk9',
    #                     '/m/0gy1t2s', '/m/02sgy', '/m/07rjwbb', '/m/06rvn', '/m/04k94', '/t/dd00077', '/m/07pggtn',
    #                     '/m/07s0s5r', '/m/07pp_mv', '/m/04rlf', '/m/0jbk', '/m/039jq', '/m/0dwsp', '/m/07q6cd_',
    #                     '/m/0239kh', '/m/0jb2l', '/m/03m9d0z', '/m/068hy', '/m/09l8g', '/m/04szw', '/m/07qnq_y',
    #                     '/m/01280g', '/m/014zdl', '/m/03w41f', '/m/0bm0k', '/m/03qtq', '/m/018vs', '/m/07yv9',
    #                     '/m/07r4wb8', '/m/0ltv', '/t/dd00134', '/m/0dwtp', '/m/07pjwq1', '/m/03wwcy', '/m/07k1x',
    #                     '/m/0395lw', '/m/026fgl', '/m/0838f', '/m/07qjznt', '/m/0f8s22', '/m/07qs1cx', '/m/0k65p',
    #                     '/m/09hlz4', '/t/dd00071', '/m/042v_gx', '/m/07pzfmf', '/m/07qn4z3', '/m/02_41', '/m/0912c9',
    #                     '/m/0bm02']

    fsd50k_black_ids = fsd50k_blacklist
    print(fsd50k_black_ids)
    print(len(fsd50k_black_ids))
    print((set(fsd50k_black_ids) == set(fsd50k_blacklist)))

    # t = dict()
    # for key, value in fsd50k_select_ids.items():
    #     t[key] = container.mid2index(value)
    # print(t)

    p = [15,23,26,33,42,43,50,75,84,90,91,111,119,140,177,180,181,182,198]
    import os, sys
    sys.path.insert(0, '../')
    from pytorch.data import FSD50K
    dataset_dir = '/import/c4dm-datasets/FSD50K'
    audio_dirs = [os.path.join(dataset_dir, 'dev_audio'), os.path.join(dataset_dir, 'eval_audio')]
    csv_paths = [os.path.join(dataset_dir, 'ground_truth', 'dev.csv'),
                 os.path.join(dataset_dir, 'ground_truth', 'eval.csv')]
    trainset = FSD50K(
        audio_dir=audio_dirs,
        csv_path=csv_paths,
        vocabulary_path=os.path.join(dataset_dir, 'ground_truth', 'vocabulary.csv'),
        clips_dir='/import/c4dm-datasets/fsd50k_clips',
        mode='fewshot',
        num_class=200,
        sample_rate=22050,
        overwrite_clips=False
    )
    cover = set(fsd50k_blacklist) | set(fsd50k_select_ids['base'])
    for instance in trainset.indices:
        if 3 in trainset.meta[instance].nonzero()[0].tolist():
            print(f"{instance}: {(set(trainset.meta[instance].nonzero()[0].tolist()) - cover)}")
