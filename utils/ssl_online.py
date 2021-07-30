import torch
import torch.nn.functional as F
from pytorch_lightning import Callback


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """SSL online evaluator"""

    def __init__(self, in_features, ft_datasets, n_splits=100):
        super().__init__()
        self.in_features = in_features
        self.ft_datasets = ft_datasets if isinstance(ft_datasets, (list, tuple)) else [ft_datasets]
        self.n_splits = n_splits
        self.features = None
        self.labels = None

    def on_validation_start(self, trainer, pl_module):
        # save train/test features and labels for each fine-tuning datasets
        self.features = [[] for _ in range(2 * len(self.ft_datasets))]
        self.labels = [[] for _ in range(2 * len(self.ft_datasets))]

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x = batch[0].to(pl_module.device)
        y = batch[1].to(pl_module.device)

        with torch.no_grad():
            z = pl_module(x)

        if trainer.distributed_backend in ('ddp', 'ddp_spawn', 'ddp2'):
            z = pl_module.all_gather(z).view(-1, z.size(-1))
            y = pl_module.all_gather(y).view(-1)

        self.features[dataloader_idx].append(z)
        self.labels[dataloader_idx].append(y)

    def on_validation_epoch_end(self, trainer, pl_module):
        for i, name in enumerate(self.ft_datasets):
            X_train = torch.cat(self.features[2*i])
            Y_train = torch.cat(self.labels[2*i])
            X_test = torch.cat(self.features[2*i+1])
            Y_test = torch.cat(self.labels[2*i+1])

            test_acc = self._compute_accuracy(X_train, Y_train, X_test, Y_test)
            pl_module.log('test_acc_{}'.format(name), test_acc)

    def _compute_accuracy(self, X_train, Y_train, X_test, Y_test):
        X_train = F.normalize(X_train)
        X_test = F.normalize(X_test)

        n_splits = X_train.size(0) if self.n_splits == -1 else self.n_splits  # GPU memory may be limited

        corrects = 0
        for X, Y in zip(X_test.split(n_splits), Y_test.split(n_splits)):
            scores = torch.einsum('ik, jk -> ij', X, X_train)  # cosine distance
            preds = Y_train[scores.argmax(1)]
            corrects += (preds == Y).long().sum().item()
        test_acc = corrects / X_test.size(0)
        return test_acc
