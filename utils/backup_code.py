import os
import shutil

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only


class BackupCode(Callback):
    def __init__(self, excludes=()):
        super().__init__()
        self.excludes = excludes

    @rank_zero_only
    def setup(self, trainer, pl_module, stage: str):
        log_dir = trainer.logger.log_dir

        # save current code except excluded files
        code_dir = os.path.join(log_dir, 'code')
        if not os.path.exists(code_dir):
            os.makedirs(code_dir, exist_ok=True)
            for fn in os.listdir(os.getcwd()):
                if not (fn[0] in ('.', '_') or fn in self.excludes):
                    copy_func = shutil.copytree if os.path.isdir(fn) else shutil.copy
                    copy_func(fn, os.path.join(code_dir, fn))
