import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np

################################
# BayesBiNN optimizer
################################
required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


class BayesBiNN(Optimizer):
    """BayesBiNN. It uses the mean-field Bernoulli approximation. Note that currently this
        optimizer does **not** support multiple model parameter groups. All model
        parameters must use the same optimizer parameters.

        model (nn.Module): network model
        train_set_size (int): number of data samples in the full training set
        lr (float, optional): learning rate
        betas (float, optional): coefficient used for computing
            running average of gradients
        prior_lamda (FloatTensor, optional): lamda of prior distribution (posterior of previous task)
            (default: None)
        num_samples (float, optional): number of MC samples
            (default: 1), if num_samples=0, we just use the point estimate mu instead of sampling
        temperature (float): temperature value of the Gumbel soft-max trick
        reweight: reweighting scaling factor of the KL term

    """

    def __init__(self, model, train_set_size, lr=1e-9, betas=0.0, prior_lamda=None, num_samples=5, lamda_init=10, lamda_std=0, temperature=1, reweight=1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if prior_lamda is not None and not torch.is_tensor(prior_lamda):
            raise ValueError(
                "Invalid prior mu value (from previous task): {}".format(prior_lamda))

        if not 0.0 <= betas < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas))
        if train_set_size < 1:
            raise ValueError(
                "Invalid number of training data points: {}".format(train_set_size))

        defaults = dict(lr=lr, beta=betas, prior_lamda=prior_lamda,
                        num_samples=num_samples, train_set_size=train_set_size, temperature=temperature, reweight=reweight)

        # only support a single parameter group.
        super(BayesBiNN, self).__init__(model.parameters(), defaults)

        self.train_modules = []
        # to obtain the trained modules in the model
        self.set_train_modules(model)

        defaults = self.defaults

        # We only support a single parameter group
        parameters = self.param_groups[0][
            'params']  # self.param_groups is a self-contained parameter group inside optimizer

        self.param_groups[0]['lr'] = lr

        device = parameters[0].device

        p = parameters_to_vector(self.param_groups[0]['params'])

        # natural parameter of Bernoulli distribution.
        mixtures_coeff = torch.randint_like(p, 2)
        # torch.log(1+p_value) - torch.log(1-p_value)  #torch.randn_like(p) # 100*torch.randn_like(p) # math.sqrt(train_set_size)*torch.randn_like(p)  #such initialization is empirically good, others are OK of course
        self.state['lamda'] = mixtures_coeff * (lamda_init + np.sqrt(lamda_std) * torch.randn_like(
            p)) + (1-mixtures_coeff) * (-lamda_init + np.sqrt(lamda_std) * torch.randn_like(p))

        # expecttion parameter of Bernoulli distribution.
        if defaults['num_samples'] <= 0:  # BCVI-D optimizer
            self.state['mu'] = torch.tanh(self.state['lamda'])
        else:
            self.state['mu'] = torch.tanh(self.state['lamda'])

        # momentum term

        self.state['momentum'] = torch.zeros_like(p, device=device)  # momentum

        # expectation parameter of prior distribution.
        if torch.is_tensor(defaults['prior_lamda']):
            self.state['prior_lamda'] = defaults['prior_lamda'].to(device)
        else:
            self.state['prior_lamda'] = torch.zeros_like(p, device=device)

        # step initilization
        self.state['step'] = 0
        self.state['temperature'] = temperature
        self.state['reweight'] = reweight

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss without doing the backward pass
        """
        if closure is None:
            raise RuntimeError(
                'For now, BayesBiNN only supports that the model/loss can be reevaluated inside the step function')

        self.state['step'] += 1

        defaults = self.defaults
        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        momentum_beta = defaults['beta']

        mu = self.state['mu']
        lamda = self.state['lamda']

        temperature = defaults['temperature']
        reweight = self.state['reweight']

        grad_hat = torch.zeros_like(lamda)

        loss_list = []
        pred_list = []

        if defaults['num_samples'] <= 0:
            # Simply using the point estimate mu instead of sampling
            w_vector = torch.tanh(self.state['lamda'])
            vector_to_parameters(w_vector, parameters)

            # Get loss and predictions
            loss, preds = closure()

            pred_list.append(preds)

            # compute the gradients over the
            linear_grad = torch.autograd.grad(loss, parameters)
            loss_list.append(loss.detach())

            grad = parameters_to_vector(linear_grad).detach()

            grad_hat = defaults['train_set_size'] * grad

        else:
            # Using Monte Carlo samples
            # sampling samples to estimate the gradients
            for _ in range(defaults['num_samples']):
                # Sample a parameter vector:
                raw_noise = torch.rand_like(mu)

                rou_vector = torch.log(
                    raw_noise/(1 - raw_noise))/2 + self.state['lamda']

                w_vector = torch.tanh(rou_vector/temperature)

                vector_to_parameters(w_vector, parameters)

                # Get loss and predictions
                loss, preds = closure()

                pred_list.append(preds)

                # compute the gradients over the
                linear_grad = torch.autograd.grad(loss, parameters)
                loss_list.append(loss.detach())

                # Convert the parameter gradient to a single vector.
                grad = parameters_to_vector(linear_grad).detach()

                scale = (1 - w_vector*w_vector+1e-10)/temperature / \
                    (1-self.state['mu']*self.state['mu']+1e-10)
                grad_hat.add_(scale * grad)

            grad_hat = grad_hat.mul(
                defaults['train_set_size'] / defaults['num_samples'])

        # Add momentum
        self.state['momentum'] = momentum_beta * self.state['momentum'] + (1-momentum_beta)*(
            grad_hat + reweight*(self.state['lamda'] - self.state['prior_lamda']))

        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))

        # Bias correction of momentum as adam
        bias_correction1 = 1 - momentum_beta ** self.state['step']

        # Update lamda vector
        self.state['lamda'] = self.state['lamda'] - \
            self.param_groups[0]['lr'] * \
            self.state['momentum']/bias_correction1

        self.state['mu'] = torch.tanh(lamda)

        return loss, pred_list
