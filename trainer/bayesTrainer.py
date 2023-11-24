from .closureTrainer import ClosureTrainer


class BayesTrainer(ClosureTrainer):
    """Extended Trainer class to cover the special case of BayesBiNN

    Necessity to have a different training function to implement mu and lambda properly 

    Args:
        Trainer (Trainer): Trainer class to extend
        *args: Variable length argument list (for the Trainer class)
        **kwargs: Arbitrary keyword arguments (most likely optimizer or scheduler parameters)
    """

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self.optimizer.update_prior_lambda()
