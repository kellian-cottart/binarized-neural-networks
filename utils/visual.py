import os
import matplotlib.pyplot as plt
import datetime
import torch
from matplotlib.ticker import AutoMinorLocator
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def versionning(folder, title, format=".pdf"):
    os.makedirs(folder, exist_ok=True)
    # YYYY-MM-DD-title-version
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    version = 1
    # while there exists a file with the same name
    while os.path.exists(os.path.join(folder, f"{timestamp}-{title}-v{version}"+format)):
        version += 1
    versionned = os.path.join(folder, f"{timestamp}-{title}-v{version}")
    return versionned + format


def visualize_sequential(title, l_accuracies, folder, epochs=None):
    """Visualize the accuracy of each task at each epoch

    Args:
        title (str): title of the figure
        l_accuracies (list): list of list of accuracies for each task at each epoch
        folder (str): folder to save the figure
        epochs (int or list): number of epochs for each task
    """
    ### CREATE FIGURE ###
    plt.figure()
    plt.xlim(0, len(l_accuracies[0])-1)
    plt.xlabel('Epochs [-]')
    plt.ylabel('Test Accuracies [%]')
    plt.ylim(0, 100)

    # Set minor ticks
    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    # major ticks every 0.1
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))

    ### COMPUTE MEAN AND STD ###
    # Transform the list of list of accuracies into a tensor of tensor of accuracies
    l_accuracies = torch.tensor(l_accuracies).detach().cpu()
    mean_accuracies = l_accuracies.mean(dim=0)*100
    std_accuracies = l_accuracies.std(dim=0)*100

    # Compute the average accuracy of all tasks
    average_accuracies = mean_accuracies.mean(dim=1)

    ### PLOT ###
    # Plot the mean accuracy
    plt.plot(range(len(mean_accuracies)), mean_accuracies)

    # Plot the average accuracy with end total accuracy
    plt.plot(range(len(average_accuracies)),
             average_accuracies, color='black', linestyle='--', zorder=0, linewidth=0.75)

    # Fill between only accepts 1D arrays for error, we need to extract each std individually
    upper_bound_tasks, lower_bound_tasks = [], []
    for task in range(len(std_accuracies[0])):
        upper_bound_tasks.append(
            mean_accuracies[:, task] + std_accuracies[:, task])
        lower_bound_tasks.append(
            mean_accuracies[:, task] - std_accuracies[:, task])
    # Plot the std accuracy
    for i in range(len(upper_bound_tasks)):
        plt.fill_between(range(len(mean_accuracies)),
                         upper_bound_tasks[i], lower_bound_tasks[i], alpha=0.2)
    # Vertical lines to separate tasks
    for i in range(1, len(l_accuracies[0][0])):
        n_epochs_task = epochs[i-1] if isinstance(epochs, list) else epochs
        plt.axvline(x=i*n_epochs_task-1, color='grey',
                    linestyle='--', linewidth=0.5, zorder=0)

    # legend is the number of the task - Accuracy of the end of this task - accuracy at the end of all tasks
    legend = []
    for i in range(len(mean_accuracies[0])):
        index = sum(epochs[:i+1])-1 if isinstance(epochs,
                                                  list) else (i+1)*epochs-1
        end = sum(epochs)-1 if isinstance(epochs,
                                          list) else len(mean_accuracies)-1
        legend += [
            f"Task {i}: Epoch {index}: {mean_accuracies[index, i]:.2f}% - Epoch {end}: {mean_accuracies[-1, i]:.2f}%"]

    legend += [f"Average of tasks: {average_accuracies[-1]:.2f}%"] + \
        ["Task change"]

    ### LEGEND ###
    plt.legend(
        legend,
        loc="lower right",
        prop={'size': 6},
    )

    ### SAVE ###
    # PDF
    versionned = versionning(folder, title, ".pdf")
    plt.savefig(versionned, bbox_inches='tight')
    # SVG
    versionned = versionning(folder, title, ".svg")
    plt.savefig(versionned, bbox_inches='tight')


def visualize_task_frame(title, l_accuracies, folder, t_start, t_end):
    """ Visualize the accuracy of each task between t_start and t_end

    Args:
        title (str): title of the figure
        l_accuracies (list): list of list of accuracies for each task at each epoch
        folder (str): folder to save the figure
        t_start (int): start task
        t_end (int): end task (included)
    """
    # Compute the number of epochs
    n_epochs = len(l_accuracies[0]) // len(l_accuracies[0][0])

    # l_accuracies is a vector of n network accuracies
    # l_accuracies[0] is the accuracy of the first network for each task
    l_accuracies = torch.tensor(l_accuracies).detach().cpu()
    mean_acc = torch.mean(l_accuracies, dim=0)
    std_acc = l_accuracies.std(dim=0)

    # Get the last epochs at t_end
    final_epoch = t_end * n_epochs - 1
    mean_acc = mean_acc[final_epoch]
    std_acc = std_acc[final_epoch]

    # Retrieve only the tasks between t_start and t_end
    mean_acc = mean_acc[t_start-1:t_end+1] * 100
    std_acc = std_acc[t_start-1:t_end+1] * 100

    plt.figure(figsize=(6, 3))
    # Scatter with line
    plt.plot(range(len(mean_acc)), mean_acc, zorder=3,
             label="BSU", marker='o', color='purple')
    # Fill between std
    plt.fill_between(range(len(mean_acc)), mean_acc-std_acc,
                     mean_acc+std_acc, alpha=0.3, zorder=2, color='purple')
    # Add MNIST baseline
    # plt.axhline(y=98.2, color='blue', linestyle='--',
    #             linewidth=1, label="MNIST baseline")
    plt.legend(loc="lower right")

    plt.xlim(t_start, t_end)
    plt.xlabel('Task [-]')
    plt.ylabel('Accuracy [%]')
    # xtickslabels from t_start to t_end as string
    plt.xticks(torch.arange(0, t_end+1-t_start).detach().cpu(),
               [str(i) for i in range(t_start, t_end+1)])
    plt.xlim(0, t_end-t_start)
    plt.ylim(0, 100)
    # increase the size of the ticks
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.tick_params(axis='y', which='minor', length=2)

    ### SAVE ###
    # PDF
    versionned = versionning(folder, title, ".pdf")
    plt.savefig(versionned, bbox_inches='tight')
    # SVG
    versionned = versionning(folder, title, ".svg")
    plt.savefig(versionned, bbox_inches='tight')


def visualize_weights(title, weights, folder):
    """Visualize the weights of the model to assess the consolidation of the knowledge
    """
    ### CREATE FIGURE ###
    plt.figure()
    ### CONVERT STATE DICT TO TENSOR ###
    tensor = torch.cat([torch.flatten(w)
                       for w in weights.values()]).detach().cpu()
    ### PLOT ###
    hist = torch.histc(tensor, bins=1000, min=-1, max=1).detach().cpu()
    plt.plot(torch.linspace(-1, 1, 1000).detach().cpu(),
             hist * 100 / len(tensor))

    plt.xlabel('Value of weights')
    plt.ylabel('% of weights')
    plt.savefig(versionning(folder, title, ".pdf"), bbox_inches='tight')
    plt.close()


def visualize_lr(lr):
    """ If the learning rate isn't local, we want to visualize the distribution of the learning rate
    """
    hist_lr = torch.histc(lr, bins=1000).detach().cpu() * 100 / len(lr)
    plt.plot(torch.linspace(0, 1e-1, 1000).detach().cpu(), hist_lr)
    plt.xlabel('Value of learning rate')
    plt.ylabel('% of learning rate')
    plt.show()


def visualize_grad(parameters, grad, path, threshold=10, task=None, epoch=None):
    """ Plot a graph with the distribution in lambda values with respect to certain thresholds

    Args:
        parameters (dict): Parameters of the model
        grad (torch.Tensor): grad values
        path (str): Path to save the graph
        threshold (int): Threshold to plot the distribution
        task (int): Task number
        epoch (int): Epoch number
    """

    params = [torch.zeros_like(param)
              for param in parameters if param.requires_grad]
    vector_to_parameters(
        grad, params)

    # figure with as many subplots as lambdas
    fig, ax = plt.subplots(len(params), 1, figsize=(5, 5*len(params)))
    for i, grad in enumerate(params):

        title = r'$\alpha \times s \odot g$' + \
            f"[{'x'.join([str(s) for s in grad.shape][::-1])}]"

        bins = 50
        hist = torch.histc(grad, bins=bins, min=-threshold,
                           max=threshold).detach().cpu()

        length = torch.prod(torch.tensor(grad.shape)).item()
        # x is the value between - threshold and -10^-10 and 10^-10 and threshold
        x = torch.cat([torch.linspace(-threshold, -1e-10, bins//2),
                       torch.linspace(1e-10, threshold, bins//2)]).detach().cpu()

        ax[i].bar(x,
                  hist * 100 / length,
                  width=2*threshold/bins,
                  zorder=2,
                  color='purple')

        ax[i].set_xscale('symlog')

        # write on the graph the maximum value and the minimum value
        ax[i].text(0.5, 0.95, f"Max: {grad.max():.2f}",
                   fontsize=6, ha='center', va='center', transform=ax[i].transAxes)
        ax[i].text(0.5, 0.9, f"Min: {grad.min():.2f}",
                   fontsize=6, ha='center', va='center', transform=ax[i].transAxes)

        ax[i].set_xlabel(r'$\alpha \times s \odot g$ [-]')
        ax[i].set_ylabel(r'Histogram of $\alpha \times s \odot g$ [%]')
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].tick_params(which='both', width=1)
        ax[i].tick_params(which='major', length=6)
        ax[i].set_ylim(0, 100)
        ax[i].set_title(title, fontsize=8)

    os.makedirs(path, exist_ok=True)
    fig.savefig(versionning(path, f"grad-task{task}-epoch{epoch}" if epoch is not None else "grad",
                ".pdf"), bbox_inches='tight')
    plt.close()


def visualize_lambda(parameters, lambda_, path, threshold=10, task=None, epoch=None):
    """ Plot a graph with the distribution in lambda values with respect to certain thresholds

    Args:
        parameters (dict): Parameters of the model
        lambda_ (torch.Tensor): Lambda values
        path (str): Path to save the graph
        threshold (int): Threshold to plot the distribution
        task (int): Task number
        epoch (int): Epoch number
    """

    params = [torch.zeros_like(param)
              for param in parameters if param.requires_grad]
    vector_to_parameters(
        lambda_, params)

    # figure with as many subplots as lambdas
    fig, ax = plt.subplots(len(params), 1, figsize=(5, 5*len(params)))
    for i, lbda in enumerate(params):

        title = r"$\lambda$" + \
            f"[{'x'.join([str(s) for s in lbda.shape][::-1])}]"

        bins = 50
        hist = torch.histc(lbda, bins=bins, min=-threshold,
                           max=threshold).detach().cpu()

        length = torch.prod(torch.tensor(lbda.shape)).item()
        # plot the histogram
        ax[i].bar(torch.linspace(-threshold, threshold, bins).detach().cpu(),
                  hist * 100 / length,
                  width=2*threshold/bins,
                  zorder=2)
        ax[i].set_xlabel('$\lambda$ [-]')
        ax[i].set_ylabel('Histogram of $\lambda$ [%]')
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].tick_params(which='both', width=1)
        ax[i].tick_params(which='major', length=6)
        ax[i].set_ylim(0, 100)
        ax[i].set_title(title, fontsize=8)

        textsize = 6
        transform = ax[i].transAxes
        ax[i].text(0.5, 0.95, f"$\lambda$  Lambda values above {threshold}: {(lbda > threshold).sum() * 100 / length:.2f}%",
                   fontsize=textsize, ha='center', va='center', transform=transform)
        ax[i].text(0.5, 0.9, f"$\lambda$ values above 2: {((lbda > 2) & (lbda < threshold)).sum() * 100 / length:.2f}%",
                   fontsize=textsize, ha='center', va='center', transform=transform)
        ax[i].text(0.5, 0.85, f"$\lambda$  values below -2: {((lbda < -2) & (lbda > -threshold)).sum() * 100 / length:.2f}%",
                   fontsize=textsize, ha='center', va='center', transform=transform)
        ax[i].text(0.5, 0.8, f"$\lambda$ values below -{threshold}: {(lbda < -threshold).sum() * 100 / length:.2f}%",
                   fontsize=textsize, ha='center', va='center', transform=transform)
        ax[i].text(0.5, 0.75, f"$\lambda$ values between -2 and 2: {((lbda < 2) & (lbda > -2)).sum() * 100 / length:.2f}%",
                   fontsize=textsize, ha='center', va='center', transform=transform)

    os.makedirs(path, exist_ok=True)
    fig.savefig(versionning(path, f"lambda-task{task}-epoch{epoch}" if epoch is not None else "lambda",
                ".pdf"), bbox_inches='tight')
    plt.close()


def visualize_certainty(predicted, certainty, path, log=True):
    """ Visualize the certainty of the model with respected to correct and incorrect predictions

    Args:
        predicted (torch.Tensor): Predicted labels
        certainty (torch.Tensor): Certainty of the model
        path (str): Path to save the graph
        log (bool): Input data is in log scale
    """
    if log:
        certainty = torch.exp(certainty)

    # We are plotting an histogram of the certainty of the model with respect to the correct and incorrect predictions
    # Correct predictions in blue and incorrect predictions in red
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins = 50
    hist_correct = torch.histc(
        certainty[predicted == 1], bins=bins, min=0, max=1).detach().cpu()
    hist_incorrect = torch.histc(
        certainty[predicted == 0], bins=bins, min=0, max=1).detach().cpu()
    ax.bar(torch.linspace(0, 1, bins).detach().cpu(),
           hist_correct * 100 / len(certainty[predicted == 1]),
           width=1/bins,
           zorder=2,
           color='blue',
           alpha=0.5)
    ax.bar(torch.linspace(0, 1, bins).detach().cpu(),
           hist_incorrect * 100 / len(certainty[predicted == 0]),
           width=1/bins,
           zorder=2,
           color='red',
           alpha=0.5)
    ax.set_xlabel('Certainty [-]')
    ax.set_ylabel('Histogram of certainty [%]')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    ax.set_ylim(0, 100)

    ax.legend(["Correct predictions", "Incorrect predictions"],
              prop={'size': 8})

    os.makedirs(path, exist_ok=True)
    fig.savefig(versionning(path, "certainty", ".pdf"), bbox_inches='tight')
    plt.close()
