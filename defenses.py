import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint


def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler,
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)

    # init delta (adv. perturbation) - FILL ME
    delta = None

    # total number of updates - FILL ME
    epochs = int(epochs/m)

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    for epoch in range(epochs):
        for i, data in enumerate(loader_tr, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            if delta is None:
                delta = torch.zeros_like(data[0])
                delta = delta.to(device)
            for j in range(m):
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                noisy_x = inputs + delta[:inputs.shape[0]]
                noisy_x = torch.clamp(noisy_x, 0, 1)
                noisy_x.requires_grad = True
                outputs = model(noisy_x)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # update delta
                delta[:inputs.shape[0]] = delta[:inputs.shape[0]] + \
                    noisy_x.grad.sign() * eps
                delta = torch.clamp(delta, -eps, eps)
                if (j+(i+epoch*len(loader_tr))*m) % scheduler_step_iters == 0:
                    lr_scheduler.step()
    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        counts = torch.zeros(4, dtype=int).to(device=x.device)
        x = x.repeat(min(batch_size, n), 1, 1, 1)
        for i in range(int(np.ceil(n/batch_size))):
            if i == int(np.ceil(n/batch_size))-1:
                x = x[:n % batch_size]
            # add noise to x - FILL ME
            noisy_x = x + \
                torch.normal(mean=0, std=self.sigma,
                             size=x.shape).to(device=x.device)

            # classify x - FILL ME
            dist = self.model(noisy_x)
            classes = dist.argmax(dim=1)

            # update class counts - FILL ME
            class_counts = torch.bincount(
                classes, minlength=4)
            counts += class_counts
        return counts

    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        with torch.no_grad():
            # find prediction (top class c) - FILL ME
            class_counts = self._sample_under_noise(x, n0, batch_size)
            c = class_counts.argmax().item()
            class_counts = self._sample_under_noise(x, n, batch_size)
            class_counts = class_counts.cpu()
            # compute lower bound on p_c - FILL ME
            ci_low, _ = proportion_confint(class_counts[c], n, 1-alpha)
            # done
            if ci_low > 0.5:
                radius = norm.ppf(ci_low) * self.sigma
                return c, radius
            return self.ABSTAIN, self.ABSTAIN


class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        mask = torch.rand(self.dim).to(device)
        trigger = torch.rand(self.dim).to(device)
        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        for _ in range(self.niters):
            for x, _ in data_loader:
                x = x.to(device)
                mask.requires_grad = True
                trigger.requires_grad = True
                inputs = (1-mask)*x + mask*trigger
                outputs = self.model(inputs)
                c_t_ = c_t*torch.ones(inputs.shape[0], dtype=int).to(device)
                loss = self.loss_func(outputs, c_t_)
                loss.backward()
                # update mask and trigger
                with torch.no_grad():
                    mask = mask - self.step_size * mask.grad.sign()
                    trigger = trigger - self.step_size * trigger.grad.sign()
                    # project mask and trigger to [0,1]
                    mask = torch.clamp(mask, 0, 1)
                    trigger = torch.clamp(trigger, 0, 1)

        # done
        return mask, trigger
