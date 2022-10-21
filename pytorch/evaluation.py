""" This module is used for evaluate the performance of modules."""
import torch
import matplotlib.pyplot as plt
from sklearn import metrics


class Metrics(object):
    def __init__(self, single_target: bool=True):
        """ Evaluate the trained model and output a series of metrics based on the task.
            Macro average is used if applicable by default.
        :param single_target: bool, determine the metrics used for evaluation according to the task
        """
        self.single_target = single_target


    def summary(self, target, prediction):
        """ Summary a series of metrics between the input prediction and corresponding target.
        :param target: list
        :param prediction: list
        :return: metrics, list
        """
        assert len(target) == len(prediction)
        if self.single_target:
            acc = calculate_acc(target, prediction)
            f1_score = calculate_f1_score(target, prediction, average='macro')
            metric_dict = {'acc': acc, 'f1_score': f1_score}
        else:
            mean_average_precision = calculate_map(target, prediction, average='macro')
            roc_auc_score = calculate_roc_auc_score(target, prediction, average='macro')
            metric_dict = {'map': mean_average_precision, 'roc_auc_score': roc_auc_score}
            # metric_dict = {'map': mean_average_precision}
        return metric_dict


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
    return metrics.f1_score(labels, predictions, average)


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


def plot_tsne(output_path: str, model: torch.nn.Module, input: torch.Tensor, target: torch.Tensor) -> None:
    """ Compute t-SNE result and plot the fig."""
    fig = plt.figure()
    ax = fig.add_subplot()

    model.eval()
    with torch.no_grad():
            output = model(input).detach().cpu().numpy()
            # Calculate t-SNE
            embeddings = TSNE(n_components=2, perplexity=10.0, early_exaggeration=100.0, learning_rate=100, n_iter=1000,
                           n_iter_without_progress=300, metric='euclidean', init='random', verbose=0,
                           random_state=42, method='barnes_hut', angle=0.5, square_distances='legacy').fit_transform(output)
            plot_markers = ['o', 'x', '+', '^', '.']
            for level in range(target.size(dim=-1)):
                max_idx = torch.max(target[:, level])
                for idx in range(max_idx):
                    plots = embeddings[target[:, level].detach().cpu().numpy() == idx]
                    ax.scatter(plots[:, 0], plots[:, 1], marker=plot_markers[level], alpha=0.5)
            fig.savefig(output_path)