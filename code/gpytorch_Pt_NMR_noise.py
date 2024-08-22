import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from generate_descriptors import generate_descriptors
# Generate training data

descriptor_path = '/home/alex/Pt_NMR/data/representations/APE_RF/'
XYZ_directory = '/home/alex/Pt_NMR/data/structures/'
descriptor_params = [3.0, 1000]
central_atom = 'Pt'

target_name = 'Experimental'
target_path = '/home/alex/Pt_NMR/data/labels/final_data_corrected'


X_data = generate_descriptors(descriptor_params=descriptor_params, descriptor_path=descriptor_path,
                              xyz_path=XYZ_directory, xyz_base='st_',
                              central_atom=central_atom).get_APE_RF()

X_data = torch.tensor(X_data)

target_data = pd.read_csv(f'{target_path}.csv')[str(target_name)]

train_x, test_x, \
            train_y, test_y = train_test_split(X_data, target_data,
                                                         random_state=42, test_size=0.25, shuffle=True)

train_y = torch.tensor(train_y.values.astype(float).flatten())
test_y = torch.tensor(test_y.values.astype(float).flatten())

# Define the GP model
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = 50000
model = GPRegressionModel(train_x, train_y, likelihood)

# Set the model in training mode
model.train()
likelihood.train()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

# Define the loss function (marginal log likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

prev_loss = float('inf')
tol = 1e-6  # Tolerance for convergence

training_iterations = 100000
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

    # Print the loss and current noise level
    noise = likelihood.noise.item()
    print(f'Iteration {i + 1}/{training_iterations} - Loss: {loss.item():.6f} - Noise: {noise:.6f}')

    # Check convergence
    if abs(prev_loss - loss.item()) < tol:
        print(f"Convergence achieved!, Noise: {noise}")
        break

    prev_loss = loss.item()

# Set the model in evaluation mode
#model.eval()
#likelihood.eval()

# Make predictions
#test_x = torch.linspace(0, 1, 51)
#with torch.no_grad(), gpytorch.settings.fast_pred_var():
#    observed_pred = likelihood(model(test_x))

# Plot the results
#with torch.no_grad():
    # Initialize plot
#    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
#    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Predictive mean as blue line
#    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')

    # Shade in confidence
#    ax.fill_between(test_x.numpy(),
#                    observed_pred.confidence_region()[0].numpy(),
#                    observed_pred.confidence_region()[1].numpy(),
#                    alpha=0.5)

#    ax.set_ylim([-3, 3])
#    ax.legend(['Observed Data', 'Mean', 'Confidence'])
#    plt.show()
