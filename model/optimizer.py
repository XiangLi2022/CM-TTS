import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer_fs2"]["betas"],
            eps=train_config["optimizer_fs2"]["eps"],
            weight_decay=train_config["optimizer_fs2"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer_fs2"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer_fs2"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer_fs2"]["anneal_rate"]
        self.current_step = current_step
        self.last_lr = self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

    def get_last_lr(self):
        return self.last_lr

    def step(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        self.last_lr = lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class ScheduledOptimDiff:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, args, model, train_config, model_config, current_step):

        self.model = args.model
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        if self.model == "diff_shallow":
            self.current_step -= train_config["step"]["total_step_aux"]
        if self.model == "diff_aux":
            self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        elif self.model in ["diff_naive", "diff_shallow", "consistency_training", "consistency_distillation"]:
            self.init_lr = train_config["optimizer"]["init_lr"]
        self.last_lr = self.init_lr

    def get_last_lr(self):
        return self.last_lr

    def step_and_update_lr(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        self.last_lr = lr
        return lr
    
    def step(self):
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        # print("self.init_lr:", self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        if self.model == "diff_aux":
            lr = np.min(
                [
                    np.power(self.current_step, -0.5),
                    np.power(self.n_warmup_steps, -1.5) * self.current_step,
                ]
            ) * self.init_lr
            for s in self.anneal_steps:
                if self.current_step > s:
                    lr = lr * self.anneal_rate
        elif self.model in ["diff_naive", "diff_shallow", "consistency_training", "consistency_distillation"]:
            lr = self.init_lr
            for s in self.anneal_steps:
                if self.current_step > s:
                    lr = lr * self.anneal_rate
        else:
            lr = self.init_lr
            for s in self.anneal_steps:
                if self.current_step > s:
                    lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr

