import sys
import torch
import pickle as pkl
import numpy as np
from torch import Tensor

sys.path.insert(1, '../')
from src.fsd_meta import FSD50K_MetaContainer


class AudioSetTaxonomy(object):
    def __init__(self, taxonomy_path: str, vocabulary_path: str) -> None:
        container = FSD50K_MetaContainer(as_taxonomy_path=taxonomy_path, fsd_vocabulary_path=vocabulary_path)
        self.taxonomy = container.as_taxonomy
        self.paths = self.traverse_paths()

    def traverse_paths(self):
        res = list()
        paths, pt2midnodes = list(), dict()
        # Start from leaf nodes
        path_id = 0
        for mid, attr in self.taxonomy.items():
            if attr['child_mid'] == []:
                paths.append([mid])  # e.g. path_0=[mid, ..., root]
                # e.g. pt2midnodes[current point] = [path_id, ...]
                try:
                    pt2midnodes[mid].append(path_id)
                except KeyError:
                    pt2midnodes[mid] = [path_id]

                path_id += 1

        # loop over to construct paths to root
        while True:
            path_id, _tmp_paths, _tmp_used, _tmp_pt = 0, list(), list(), dict()
            for mid, attr in self.taxonomy.items():
                # add the mid to the end of paths if the current point to its child class
                # also add this mid to the next current point set
                for _node in pt2midnodes.keys():
                    if _node in attr['child_mid']:
                        for _pid in pt2midnodes[_node]:
                            _tmp_paths.append([*paths[_pid], mid])
                            _tmp_used.append(_pid)

                            try:
                                _tmp_pt[mid].append(path_id)
                            except KeyError:
                                _tmp_pt[mid] = [path_id]

                            path_id += 1

            _complete_path = [p for id, p in enumerate(paths) if id not in _tmp_used]
            if len(_complete_path) != 0:
                res.extend(_complete_path)

            if len(_tmp_paths) == 0 and len(_tmp_pt) == 0:
                print("Paths are searched completely.")
                break
            else:
                paths, pt2midnodes= _tmp_paths, _tmp_pt

        return res

    def get_taxonomy_distance(self, mids):
        """ Find the lowest common ancestor of a pair of classes."""
        m1, m2 = mids
        p1, p2 = list(), list()

        for _p in self.paths:
            for _lvl in range(len(_p)):
                if _p[_lvl] == m1:
                    p1.append(_p[_lvl:])

                if _p[_lvl] == m2:
                    p2.append(_p[_lvl:])

        distances = list()
        for _p1 in p1:
            for _p2 in p2:
                distances.append(self.path_distance(_p1, _p2))

        return np.min(distances)

    def get_parent_mid(self, mid):
        """ Retrieve the parent mid of an input mid."""
        parent_mid = list()
        for _mid, _attr in self.taxonomy.items():
            if mid in _attr['child_mid']:
                parent_mid.append(_mid)

        return parent_mid

    @classmethod
    def path_distance(self, a, b):
        """ Measure number of intermediate edges between two array-like paths. e.g., path_0 = [leaf, ..., root]."""
        la, lb = len(a), len(b)
        a, b = np.array(a), np.array(b)
        if a.shape[0] < b.shape[0]:
            n_pad = b.shape[0] - a.shape[0]
            a = np.pad(a, (n_pad, 0), 'constant', constant_values=0)
        elif a.shape[0] > b.shape[0]:
            n_pad = a.shape[0] - b.shape[0]
            b = np.pad(b, (n_pad, 0), 'constant', constant_values=0)

        n_shared_nodes = np.count_nonzero((a == b))

        return la + lb - n_shared_nodes * 2


class label_embedding(object):
    def __init__(self, weight_path: str, n_classes: int, beta: float, trainable: bool = False) -> None:
        with open(weight_path, 'rb') as f:
            w = pkl.load(f)
        self.weight_matrix = torch.from_numpy(w)

        self.n_classes = n_classes
        self.emb_layer = torch.nn.Embedding(self.n_classes, self.n_classes)
        self.beta = beta
        self.trainable = trainable

    def __call__(self, slt_classes: list, y: Tensor) -> Tensor:
        """ Label embedding layer. `y` should be in the one-hot format = [n_embedding, embedding_dim]."""
        slt_classes = torch.tensor(slt_classes)
        assert self.n_classes == slt_classes.size(dim=0)

        # Retrieve weights as per `slt_classes`
        weights = self.weight_matrix.index_select(dim=0, index=slt_classes).index_select(dim=1, index=slt_classes).float()

        # Control the weights by hyper-param beta and re-scale the weights to [0, 1]
        weights *= -self.beta
        weights = weights.softmax(dim=1)

        self.emb_layer.load_state_dict({"weight": weights})
        if not self.trainable:
            self.emb_layer.weight.requires_grad = False

        # Adjust y into index format
        indices_y = y.nonzero(as_tuple=True)[-1]  # `embedding_dim` for dim = 1
        assert y.size(dim=0) == indices_y.size(dim=0)

        return self.emb_layer(indices_y)


if __name__ == '__main__':
    """ This script could be used for generating weight."""
    taxonomy_path = '../src/taxonomy/ontology.json'
    vocabulary_path = '../src/taxonomy/ground_truth/vocabulary.csv'
    container = FSD50K_MetaContainer(as_taxonomy_path=taxonomy_path, fsd_vocabulary_path=vocabulary_path)
    taxonomy = AudioSetTaxonomy(taxonomy_path, vocabulary_path)
    # Row denotes a specific label and col denotes another label to this label
    # if the element is -1 the distance is not available in our dataset
    distance_matrix = np.ones((200, 200), dtype=int) * (-1)
    for row in range(200):
        for col in range(200):
            distance_matrix[row][col] = taxonomy.get_taxonomy_distance([container.index2mid[row], container.index2mid[col]])

    with open('./fsd50k_distance_matrix.pkl', 'wb') as f:
        pkl.dump(distance_matrix, f)

    print(distance_matrix)

    """ Here we test some functions and the generated weight matrix."""
    slt_classes = [37, 33, 75, 198, 84]
    y =torch.tensor([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0]])
    emb_layer = label_embedding(
        weight_path='/homes/jl009/WORKSPACE/AudioTagging/src/fsd50k_distance_matrix.pkl', n_classes=5, beta=3
    )
    embedded_label = emb_layer(slt_classes=slt_classes, y=y)
    print(embedded_label)
