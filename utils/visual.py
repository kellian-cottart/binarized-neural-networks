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

    # grid but only horizontal
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

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
    # grid but only horizontal
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.xlim(t_start, t_end)
    plt.xlabel('Task')
    plt.ylabel('Accuracy %')
    # ticks every 5% accuracy
    plt.yticks(torch.arange(0, 101, 20).detach().cpu())
    # xtickslabels from t_start to t_end as string
    plt.xticks(torch.arange(0, t_end+1-t_start).detach().cpu(),
               [str(i) for i in range(t_start, t_end+1)])
    plt.xlim(0, t_end-t_start)
    plt.ylim(0, 100)
    plt.grid(True, zorder=0)
    # increase the size of the ticks
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)

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


def visualize_lambda(lambda_, path, threshold=10):
    """ Plot a graph with the distribution in lambda values with respect to certain thresholds

    Args:
        lambda_ (torch.Tensor): Lambda values
        path (str): Path to save the graph
        threshold (int): Threshold to plot the distribution
    """
    plt.figure()
    plt.grid()
    bins = 100
    hist = torch.histc(lambda_, bins=bins, min=-threshold,
                       max=threshold).detach().cpu()

    # total number of values in lambda
    shape = lambda_.shape[0] * lambda_.shape[1]

    plt.bar(torch.linspace(-threshold, threshold, bins).detach().cpu(),
            hist * 100 / shape,
            width=1.5,
            zorder=2)
    plt.xlabel('Value of $\lambda$ ')
    plt.ylabel('% of $\lambda$')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().tick_params(which='both', width=1)
    plt.gca().tick_params(which='major', length=6)
    plt.ylim(0, 100)

    textsize = 6
    transform = plt.gca().transAxes

    plt.text(0.5, 0.95, f"$\lambda$  Lambda values above {threshold}: {(lambda_ > threshold).sum() * 100 / shape:.2f}%",
             fontsize=textsize, ha='center', va='center', transform=transform)
    plt.text(0.5, 0.9, f"$\lambda$ values above 2: {((lambda_ > 2) & (lambda_ < threshold)).sum() * 100 / shape:.2f}%",
             fontsize=textsize, ha='center', va='center', transform=transform)
    plt.text(0.5, 0.85, f"$\lambda$  values below -2: {((lambda_ < -2) & (lambda_ > -threshold)).sum() * 100 / shape:.2f}%",
             fontsize=textsize, ha='center', va='center', transform=transform)
    plt.text(0.5, 0.8, f"$\lambda$ values below -{threshold}: {(lambda_ < -threshold).sum() * 100 / shape:.2f}%",
             fontsize=textsize, ha='center', va='center', transform=transform)
    plt.text(0.5, 0.75, f"$\lambda$ values between -2 and 2: {((lambda_ < 2) & (lambda_ > -2)).sum() * 100 / shape:.2f}%",
             fontsize=textsize, ha='center', va='center', transform=transform)

    os.makedirs(path, exist_ok=True)
    plt.savefig(versionning(path, "lambda-visualization",
                ".pdf"), bbox_inches='tight')
    plt.close()
