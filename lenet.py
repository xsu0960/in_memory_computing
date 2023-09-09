from torch import Tensor
from aihwkit.nn import AnalogLinear
import os


model = AnalogLinear(2, 2)
model(Tensor([[0.1, 0.2], [0.3, 0.4]]))


def create_rpu_config():
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.configs.utils import BoundManagementType, WeightNoiseType
    from aihwkit.inference import PCMLikeNoiseModel
    from aihwkit.simulator.configs import SingleRPUConfig, IOParameters

    rpu_config = InferenceRPUConfig()
    # rpu_config = SingleRPUConfig(forward=IOParameters(is_perfect=True),backward=IOParameters(is_perfect=True))
    rpu_config.backward.bound_management = BoundManagementType.NONE
    rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = 0.02
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0, read_noise_scale=0.2, drift_scale=0.2)

    return rpu_config


from torch.nn import Tanh, MaxPool2d, LogSoftmax, Flatten
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential


def create_analog_network(rpu_config):
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        Tanh(),
        MaxPool2d(kernel_size=2),
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1,
                     rpu_config=rpu_config),
        Tanh(),
        MaxPool2d(kernel_size=2),
        Tanh(),
        Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
        Tanh(),
        AnalogLinear(in_features=channel[3], out_features=10, rpu_config=rpu_config),
        LogSoftmax(dim=1)
    )

    return model


from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()

from aihwkit.optim import AnalogSGD


def create_analog_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained

    Returns:
        Optimizer: created analog optimizer
    """

    optimizer = AnalogSGD(model.parameters(), lr=0.01)  # we will use a learning rate of 0.01 as in the paper
    optimizer.regroup_param_groups(model)

    return optimizer


from torch import device, cuda

#DEVICE = device('cuda' if cuda.is_available() else 'cpu')
DEVICE = device('cpu')
print('Running the simulation on: ', DEVICE)


def train_step(train_data, model, criterion, optimizer):
    """Train network.

    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer

    Returns:
        train_dataset_loss: epoch loss of the train dataset
    """
    total_loss = 0
    print("used model train")
    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    train_dataset_loss = total_loss / len(train_data.dataset)

    return train_dataset_loss


def test_step(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        test_dataset_loss: epoch loss of the train_dataset
        test_dataset_error: error of the test dataset
        test_dataset_accuracy: accuracy of the test dataset
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        test_dataset_accuracy = predicted_ok / total_images * 100
        test_dataset_error = (1 - predicted_ok / total_images) * 100

    test_dataset_loss = total_loss / len(validation_data.dataset)

    return test_dataset_loss, test_dataset_error, test_dataset_accuracy


def plot_results(train_losses, valid_losses, test_error):
    """Plot results.
    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, 'r-s', valid_losses, 'b-o')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:2], ['Training Losses', 'Validation Losses'])
    plt.xlabel('Epoch number')
    plt.ylabel('Loss [A.U.]')
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'test_losses.png'))
    plt.close()

    fig = plt.plot(test_error, 'r-s')
    plt.title('aihwkit LeNet5')
    plt.legend(fig[:1], ['Validation Error'])
    plt.xlabel('Epoch number')
    plt.ylabel('Test Error [%]')
    plt.yscale('log')
    plt.ylim((5e-1, 1e2))
    plt.grid(which='both', linestyle='--')
    plt.savefig(os.path.join(RESULTS, 'test_error.png'))
    plt.close()


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS = os.path.join(os.getcwd(), 'drift_results', 'perfect')
os.makedirs(RESULTS, exist_ok=True)


def training_loop(model, criterion, optimizer, train_data, validation_data, epochs=10, print_every=1):
    """Training loop.

    Args:
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer
        train_data (DataLoader): Validation set to perform the evaluation
        validation_data (DataLoader): Validation set to perform the evaluation
        epochs (int): global parameter to define epochs number
        print_every (int): defines how many times to print training progress

    """
    train_losses = []
    valid_losses = []
    test_error = []

    print("Start to train model")
    # Train model
    for epoch in range(0, epochs):
        # Train_step
        print("strain epoch: ", epoch)
        train_loss = train_step(train_data, model, criterion, optimizer)
        train_losses.append(train_loss)
        print("print_every: ", print_every)

        if epoch % print_every == (print_every - 1):
            # Validate_step
            print("in the if condition")
            with torch.no_grad():
                valid_loss, error, accuracy = test_step(validation_data, model, criterion)
                valid_losses.append(valid_loss)
                test_error.append(error)

            print(f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Test error: {error:.2f}%\t'
                  f'Test accuracy: {accuracy:.2f}%\t')
    # Save results and plot figures
    model_file = 'saved_model.pth'
    torch.save(model.state_dict(), model_file)
    np.savetxt(os.path.join(RESULTS, "Test_error.csv"), test_error, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Train_Losses.csv"), train_losses, delimiter=",")
    np.savetxt(os.path.join(RESULTS, "Valid_Losses.csv"), valid_losses, delimiter=",")
    plot_results(train_losses, valid_losses, test_error)

    return model, optimizer, (train_losses, valid_losses, test_error)


from torchvision import datasets, transforms

PATH_DATASET = os.path.join('data', 'DATASET')
os.makedirs(PATH_DATASET, exist_ok=True)


def load_images():
    """Load images for train from torchvision datasets."""

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    test_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

    return train_data, test_data


import torch
from aihwkit.simulator.configs.utils import IOParameters

torch.manual_seed(1)
# load the dataset
print("load data")
train_data, test_data = load_images()
print("create rpu config")

# create the rpu_config
rpu_config = create_rpu_config()
# rpu_config = None

print("create model")
# create the model
model = create_analog_network(rpu_config).to(DEVICE)
print("create optimizer")
# define the analog optimizer
optimizer = create_analog_optimizer(model)
print("training loop start")
training_loop(model, criterion, optimizer, train_data, test_data)

# print('\nStarting Inference Pass:\n')
#
# test_inference(model, criterion, test_data)

