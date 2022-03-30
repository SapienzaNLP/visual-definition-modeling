from typing import Optional

from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class IMGDropoutLogger(Callback):
    r"""
    Automatically monitor and logs learning rate for learning rate schedulers during training.

    Args:
        logging_interval: set to `epoch` or `step` to log `lr` of all optimizers
            at the same interval, set to `None` to log at individual interval
            according to the `interval` key of each scheduler. Defaults to ``None``.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LearningRateMonitor
        >>> lr_monitor = LearningRateMonitor(logging_interval='step')
        >>> trainer = Trainer(callbacks=[lr_monitor])

    Logging names are automatically determined based on optimizer class name.
    In case of multiple optimizers of same type, they will be named `Adam`,
    `Adam-1` etc. If a optimizer has multiple parameter groups they will
    be named `Adam/pg1`, `Adam/pg2` etc. To control naming, pass in a
    `name` keyword in the construction of the learning rate schdulers

    Example::

        def configure_optimizer(self):
            optimizer = torch.optim.Adam(...)
            lr_scheduler = {'scheduler': torch.optim.lr_schedulers.LambdaLR(optimizer, ...)
                            'name': 'my_logging_name'}
            return [optimizer], [lr_scheduler]

    """

    def __init__(self, logging_interval: Optional[str] = None):
        if logging_interval not in (None, 'step', 'epoch'):
            raise MisconfigurationException(
                'logging_interval should be `step` or `epoch` or `None`.'
            )

        self.logging_interval = logging_interval
        self.lrs = None
        self.lr_sch_names = []

    def on_train_start(self, trainer, pl_module):
        """
        Called before training, determines unique names for all lr
        schedulers in the case of multiple of the same type or in
        the case of multiple parameter groups
        """

        self.encoder = pl_module.model.encoder

    def on_batch_start(self, trainer, pl_module):
        if self.logging_interval != 'epoch':
            latest_stat = {"img_dropout": self.encoder.get_img_dropout_val()}

            if trainer.logger is not None and latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.global_step)

    def on_epoch_start(self, trainer, pl_module):
        if self.logging_interval != 'step':
            latest_stat = {"img_dropout": self.encoder.get_img_dropout_val()}

            if trainer.logger is not None and latest_stat:
                trainer.logger.log_metrics(latest_stat, step=trainer.current_epoch)
