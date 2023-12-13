import os
import matplotlib.pyplot as plt
import datetime
import torch
from matplotlib.ticker import AutoMinorLocator


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


def visualize_sequential(title, l_accuracies, folder, sequential=False):
    """Visualize the accuracy of each task at each epoch

    Args:
        title (str): title of the figure
        l_accuracies (list): list of list of accuracies for each task at each epoch
        folder (str): folder to save the figure
    """
    ### CREATE FIGURE ###
    plt.figure()
    plt.xlim(0, len(l_accuracies[0])-1)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracies')
    plt.ylim(0, 1)

    # Set minor ticks
    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    # major ticks every 0.1
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    ### COMPUTE MEAN AND STD ###
    # Transform the list of list of accuracies into a tensor of tensor of accuracies
    l_accuracies = torch.tensor(l_accuracies).detach().cpu()
    mean_accuracies = l_accuracies.mean(dim=0)
    std_accuracies = l_accuracies.std(dim=0)

    ### PLOT ###
    # Plot the mean accuracy
    plt.plot(range(len(mean_accuracies)), mean_accuracies)
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
    n_epochs_task = len(l_accuracies[0]) // len(l_accuracies[0][0])
    for i in range(1, len(l_accuracies[0][0])):
        plt.axvline(x=i*n_epochs_task-1, color='k',
                    linestyle='--', linewidth=0.5)

    if sequential:
        # Legend is name of the task - Accuracy of end of task 1 - Accuracy of end of task 2
        legend = [f"MNIST - T1 End: {mean_accuracies[n_epochs_task-1, 0]*100:.2f}% - T2 End: {mean_accuracies[-1, 0]*100:.2f}%",
                  f"Fashion MNIST - T1 End: {mean_accuracies[n_epochs_task-1, 1]*100:.2f}% - T2 End: {mean_accuracies[-1, 1]*100:.2f}%",
                  "Task change"]
        plt.axhline(y=0.982, color='blue', linestyle='--', linewidth=0.75)
        plt.axhline(y=0.899, color='orange', linestyle='--', linewidth=0.75)
        legend += ["Baseline MNIST - 98.2%", "Baseline Fashion MNIST - 89.9%"]
    else:
        # legend is the number of the task - Accuracy of the end of this task - accuracy at the end of all tasks
        legend = [f"T{i+1} - T End: {mean_accuracies[(i+1)*n_epochs_task-1, i]*100:.2f}% - All End: {mean_accuracies[-1, i]*100:.2f}%" for i in range(
            len(mean_accuracies[0]))] + ["Task change"]
        plt.axhline(y=0.982, color='blue', linestyle='--', linewidth=0.75)
        legend += ["Baseline - 98.2%"]

    ### LEGEND ###
    plt.legend(
        legend,
        loc="lower right",
        prop={'size': 8 if sequential else 6},
    )

    # grid but only horizontal
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

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
