import os
import matplotlib.pyplot as plt
import datetime
import torch


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


def visualize_sequential(title, l_accuracies, folder):
    """Visualize the accuracy of each task at each epoch

    Args:
        title (str): title of the figure
        l_accuracies (list): list of list of accuracies for each task at each epoch
        folder (str): folder to save the figure
    """
    ### CREATE FIGURE ###
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    ### COMPUTE MEAN AND STD ###
    # Transform the list of list of accuracies into a tensor of tensor of accuracies
    l_accuracies = torch.tensor(l_accuracies)
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

    ### LEGEND ###
    plt.legend(
        [f"Task {i+1}" for i in range(len(l_accuracies[0][0]))] + ["Task change"])

    ### SAVE ###
    versionned = versionning(folder, title)
    plt.savefig(versionned, bbox_inches='tight')
