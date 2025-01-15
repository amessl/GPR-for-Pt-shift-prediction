import gpytorch

class gpr_estimator(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, mean_function=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_function if mean_function else gpytorch.means.ZeroMean()
        self.covar_module = kernel if kernel else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)