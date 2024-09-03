import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np

################################
# BayesBiNN optimizer
################################


class BayesBiNN(Optimizer):
    """BayesBiNN. It uses the mean-field Bernoulli approximation. Note that currently this
        optimizer does **not** support multiple model parameter groups. All model
        parameters must use the same optimizer parameters.

        model (nn.Module): network model
        train_set_size (int): number of data samples in the full training set
        lr (float, optional): learning rate
        betas (float, optional): coefficient used for computing
            running average of gradients
        prior_lambda (FloatTensor, optional): lamda of prior distribution (posterior of previous task)
            (default: None)
        num_samples (float, optional): number of MC samples
            (default: 1), if num_samples=0, we just use the point estimate mu instead of sampling
        temperature (float): temperature value of the Gumbel soft-max trick
        reweight: reweighting scaling factor of the KL term

    """

    def __init__(self, model, train_set_size, lr=1e-9, betas=0.0, prior_lambda=None, num_samples=5, lamda_init=10, lamda_std=0, temperature=1, reweight=1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if prior_lambda is not None and not torch.is_tensor(prior_lambda):
            raise ValueError(
                "Invalid prior mu value (from previous task): {}".format(prior_lambda))

        if not 0.0 <= betas < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas))
        if train_set_size < 1:
            raise ValueError(
                "Invalid number of training data points: {}".format(train_set_size))

        defaults = dict(lr=lr, beta=betas, prior_lambda=prior_lambda,
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
        self.state['lambda'] = mixtures_coeff * (lamda_init + np.sqrt(lamda_std) * torch.randn_like(
            p)) + (1-mixtures_coeff) * (-lamda_init + np.sqrt(lamda_std) * torch.randn_like(p))

        # expecttion parameter of Bernoulli distribution.
        if defaults['num_samples'] <= 0:  # BCVI-D optimizer
            self.state['mu'] = torch.tanh(self.state['lambda'])
        else:
            self.state['mu'] = torch.tanh(self.state['lambda'])

        # momentum term

        self.state['momentum'] = torch.zeros_like(p, device=device)  # momentum

        # expectation parameter of prior distribution.
        if torch.is_tensor(defaults['prior_lambda']):
            self.state['prior_lambda'] = defaults['prior_lambda'].to(device)
        else:
            self.state['prior_lambda'] = torch.zeros_like(p, device=device)

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
        lr = self.param_groups[0]['lr']
        momentum_beta = defaults['beta']
        momentum = self.state['momentum']

        mu = self.state['mu']
        lamda = self.state['lambda']

        temperature = defaults['temperature']
        reweight = self.state['reweight']

        grad_hat = torch.zeros_like(lamda)

        loss_list = []
        pred_list = []

        if defaults['num_samples'] <= 0:
            # Simply using the point estimate mu instead of sampling
            w_vector = torch.tanh(self.state['lambda'])
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
                    raw_noise/(1 - raw_noise))/2 + self.state['lambda']
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
            grad_hat + reweight*(self.state['lambda'] - self.state['prior_lambda']))

        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))

        # Bias correction of momentum as adam
        bias_correction1 = 1 - momentum_beta ** self.state['step']

        # Update lamda vector
        self.state['lambda'] = self.state['lambda'] - \
            self.param_groups[0]['lr'] * \
            self.state['momentum']/bias_correction1
        self.state['mu'] = torch.tanh(lamda)
        return loss

    def get_distribution_params(self):
        """Returns current mean and precision of variational distribution
           (usually used to save parameters from current task as prior for next task).
        """
        mu = self.state['mu'].clone().detach()
        precision = mu*(1-mu)  # variance term

        return mu, precision

    def get_mc_predictions(self, forward_function, inputs, ret_numpy=False, raw_noises=None, *args, **kwargs):
        """Returns Monte Carlo predictions.
        Arguments:
            forward_function (callable): The forward function of the model
                that takes inputs and returns the outputs.
            inputs (FloatTensor): The inputs to the model.
            ret_numpy (bool): If true, the returned list contains numpy arrays,
                otherwise it contains torch tensors.
        """

        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        predictions = []
        noise = []
        n_samples = self.param_groups[0]['num_samples']

        if n_samples > 0:
            for _ in range(n_samples):
                network_sample = torch.bernoulli(
                    torch.sigmoid(2*self.state["lambda"]))
                noise.append(2*network_sample-1)
        elif n_samples == 0:
            # If we do not want to sample, we do a bernouilli
            noise.append(2*torch.where(
                self.optimizer.state["lamda"] <= 0,
                torch.zeros_like(self.state["lambda"]),
                torch.ones_like(self.state["lambda"])
            )-1)
        else:
            # If we only take lambda as the weights
            noise.append(self.state["lambda"])

        for n in noise:
            vector_to_parameters(n, parameters)
            prediction = forward_function(inputs)
            predictions.append(prediction)
        return predictions

    def update_prior_lambda(self):
        self.state["prior_lambda"] = self.state["lambda"]
