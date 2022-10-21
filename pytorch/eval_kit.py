import os
import torch
import scipy
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from sklearn import metrics
from sklearn.manifold import TSNE

from typing import Optional, Tuple, Union
from matplotlib.lines import Line2D


class Metrics(object):
    def __init__(self, single_target: bool = True, average: str = 'macro'):
        """ Evaluate the trained model and output a series of metrics based on the task.
            Macro average is used if applicable by default.
        :param single_target: bool, determine the metrics used for evaluation according to the task
        """
        self.single_target = single_target
        self.average = average


    def summary(self, target, prediction):
        """ Summary a series of metrics between the input prediction and corresponding target.
        :param target: list
        :param prediction: list
        :return: metrics, list
        """
        assert len(target) == len(prediction)
        if self.single_target:
            acc = calculate_acc(target, prediction)
            f1_score = calculate_f1_score(target, prediction, average=self.average)
            metric_dict = {'acc': acc, 'f1_score': f1_score}
        else:
            mean_average_precision = calculate_map(target, prediction, average=self.average)
            roc_auc_score = calculate_roc_auc_score(target, prediction, average=self.average)
            threshold = 0.2
            f1_score = calculate_f1_score(target, [(p > threshold).astype(float) for p in prediction], average=self.average)
            metric_dict = {'map': mean_average_precision, 'roc_auc_score': roc_auc_score, 'f1_score': f1_score}
        return metric_dict


# class Visualiser(object):
#     """ Visualisation result of the model."""
#     def __init__(self, output_dir: str, fig: Optional[List[str]]) -> None:
#         self.output_dir = output_dir
#         self.fig = fig if fig else ['tsne']
#
#     def plot(self, ):



def calculate_acc(labels, predictions):
    """ Calculate accuracy by comparing predictions with ground truth.
    :param labels: list, class indices of outputs of a model
    :param predictions: list, class indices of outputs of a model
    """
    return metrics.accuracy_score(labels, predictions)


def calculate_f1_score(labels, predictions, average='macro'):
    """ Calculate f1 score by comparing predictions with ground truth.
    :param labels: list, class indices of outputs of a model
    :param predictions: list, class indices of outputs of a model
    """
    return metrics.f1_score(labels, predictions, average=average)


def calculate_map(labels, predictions, average='macro'):
    """ Calculate mean average precision (mAP) by comparing predictions with ground truth.
    :param labels: list, class indices of outputs of a model
    :param predictions: list, class indices of outputs of a model
    :param average: str, summarise the result statistically, default is 'macro'
    """
    return metrics.average_precision_score(labels, predictions, average=average)


def calculate_roc_auc_score(labels, predictions, average=None):
    """ Calculate roc auc score by predictions with ground truth.
    :param labels: list, class indices of outputs of a model
    :param predictions: list, class indices of outputs of a model
    :param average: str, summarise the result statistically
    """
    return metrics.roc_auc_score(labels, predictions, average=average)


def plot_tsne(output_path: str, model: torch.nn.Module, input: Tensor, targets: Tuple[Tensor, ...]) -> None:
    """ Compute t-SNE result and plot the fig.
        Args:
            output_path: path to output file, '.jpg' or '.png'
            model: pytorch model used for visualisation
            input: input data toward model
            targets: ground truth of input data. Multi-label and multi-class are supported.
                for multi-task, target should be list of sequence, e.g., [T_1, T_2]
                for multi-class, target should be in the format of onehot., i.e., target[0].size() > 1
    """
    n_sample, _ = input.size()
    for _t in targets:
        _n_sample, _ = _t.size()
        if n_sample != _n_sample:
            raise ValueError("Size of input not equal to size of target.")

    model.eval()
    with torch.no_grad():
        output = model(input).detach().cpu().numpy()

    # Calculate t-SNE
    embeddings = TSNE(n_components=2, perplexity=10.0, early_exaggeration=100.0, learning_rate=100, n_iter=1000,
                   n_iter_without_progress=300, metric='euclidean', init='random', verbose=0,
                   random_state=42, method='barnes_hut', angle=0.5, square_distances='legacy').fit_transform(output)

    head, tail = output_path.split('.')
    marker_dict = Line2D.markers
    # plot_colors = ['b', 'g', 'r', 'm', 'k']
    # plot_markers = ['1', '2', '3', '4', 'x', '+', '|', '-', '^', 'v', '<', '>']
    for tid, target in enumerate(targets):
        _, y_dim = target.size()
        single_target = True if y_dim == 1 else False
        n_class = torch.max(target) if single_target else y_dim

        fig = plt.figure()
        ax = fig.add_subplot()

        for cls_id in range(n_class):
            plots = embeddings[target.detach().cpu().numpy() == cls_id]
            ax.scatter(plots[:, 0], plots[:, 1], marker=plot_markers[level], alpha=0.5)
        fig.savefig(os.path.join(head, f"{tid}", tail))


def confidence_interval(x: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    n_sample = x.shape[0]
    mean, std = x.mean(), x.std()
    dof = n_sample - 1  # degree of freedom
    t = np.abs(scipy.stats.t.ppf((1-confidence)/2, dof))
    interval = t * std / np.sqrt(n_sample)

    return mean, interval


if __name__ == '__main__':
    x = np.asarray([0.38848810929599314, 0.3813457894793975, 0.3883685891652951, 0.405974224100056, 0.3712585248084955,
                    0.37368518812012275, 0.3846971684495603, 0.37862704773192224, 0.38424691210403616, 0.40138234550077123,
                    0.39686110554314336, 0.3711061707913712, 0.3898808512276162, 0.37094598639773896, 0.3917694150177586])
    mean, interval = confidence_interval(x=x, confidence=0.95)
    print(mean, interval)
