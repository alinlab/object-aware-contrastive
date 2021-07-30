import os
from pytorch_lightning.callbacks import Callback


class ModelCheckpoint(Callback):
    """Save model for every {save_freq} epochs"""

    def __init__(self, save_freq=1, save_weights_only=False):
        self.ckpt_dir = None
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.ckpt_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
        if not trainer.fast_dev_run and trainer.is_global_zero:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch = trainer.current_epoch + 1  # start from 1
        if epoch % self.save_freq == 0:
            ckpt_path = os.path.join(self.ckpt_dir, '{}.ckpt'.format(epoch))
            trainer.save_checkpoint(ckpt_path, self.save_weights_only)
        if epoch == trainer.max_epochs:
            ckpt_path = os.path.join(self.ckpt_dir, 'last.ckpt')
            trainer.save_checkpoint(ckpt_path, self.save_weights_only)
