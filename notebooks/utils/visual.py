import os
import matplotlib.pyplot as plt
import datetime
import torch
from matplotlib.ticker import AutoMinorLocator
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from optimizers.bhutest import BinaryHomosynapticUncertaintyTest


def graphs(data, main_folder, net_trainer, i, epoch, predictions, labels, modulo=10):
    if (epoch % modulo == modulo-1 or epoch == 0):
        # print predicted and associated certainty
        # visualize_certainty(
        #     predictions=predictions,
        #     labels=labels,
        #     path=os.path.join(main_folder, "certainty"),
        #     task=i+1,
        #     epoch=epoch+1,
        # )
        if data["optimizer"] in [BinaryHomosynapticUncertaintyTest]:
            visualize_grad(
                parameters=net_trainer.optimizer.param_groups[0]['params'],
                grad=net_trainer.optimizer.state['grad'],
                path=os.path.join(main_folder, "grad"),
                task=i+1,
                epoch=epoch+1,
            )
            visualize_lambda(
                parameters=net_trainer.optimizer.param_groups[0]['params'],
                lambda_=net_trainer.optimizer.state['lambda'],
                path=os.path.join(main_folder, "lambda"),
                threshold=10,
                task=i+1,
                epoch=epoch+1,
            )
            visualize_lr(
                parameters=net_trainer.optimizer.param_groups[0]['params'],
                lr=net_trainer.optimizer.state['lr'],
                path=os.path.join(main_folder, "lr"),
                task=i+1,
                epoch=epoch+1,
            )
        else:
            params = [
                p for p in net_trainer.optimizer.param_groups[0]['params']]
            visualize_grad(
                parameters=params,
                grad=[
                    p.grad for p in net_trainer.optimizer.param_groups[0]['params']],
                path=os.path.join(main_folder, "grad"),
                task=i+1,
                epoch=epoch+1,
            )
            visualize_lambda(
                parameters=params,
                lambda_=params,
                path=os.path.join(main_folder, "lambda"),
                threshold=10,
                task=i+1,
                epoch=epoch+1,
            )


def versionning(folder, title, format=".pdf"):
    os.makedirs(folder, exist_ok=True)
    # YYYY-MM-DD-hh-mm-ss-title-version.format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M:%S")
    version = 1
    # while there exists a file with the same name
    while os.path.exists(os.path.join(folder, f"{timestamp}-{title}-v{version}"+format)):
        version += 1
    versionned = os.path.join(folder, f"{timestamp}-{title}-v{version}")
    return versionned + format


def visualize_sequential(title, l_accuracies, folder, epochs=None, training_accuracies=None):
    """Visualize the accuracy of each task at each epoch

    Args:
        title (str): title of the figure
        l_accuracies (list): list of list of accuracies for each task at each epoch
        folder (str): folder to save the figure
        epochs (int or list): number of epochs for each task
        training_accuracies (list): list of list of training accuracies for each task at each epoch
    """
    ### CREATE FIGURE ###
    plt.figure()
    plt.xlim(0, len(l_accuracies[0])-1)
    plt.xlabel('Epochs [-]')
    plt.ylabel('Accuracies [%]')
    plt.ylim(0, 100)

    # Set minor ticks
    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)

    ### COMPUTE MEAN AND STD ###
    # Transform the list of list of accuracies into a tensor of tensor of accuracies
    l_accuracies = torch.tensor(l_accuracies).detach().cpu()
    mean_accuracies = l_accuracies.mean(dim=0)*100
    std_accuracies = l_accuracies.std(dim=0)*100

    ### PLOT ###
    # Create a gradient of len(l_accuracies[0]) colors corresponding to each task.

    if len(mean_accuracies[0]) > 2:
        colors = plt.get_cmap('viridis', len(mean_accuracies[0]))
    else:
        colors = plt.get_cmap('Spectral_r', len(mean_accuracies[0]))
    # Plot the mean accuracy
    for i in range(len(mean_accuracies[0])):
        plt.plot(range(len(mean_accuracies)),
                 mean_accuracies[:, i], color=colors(i))

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
                         upper_bound_tasks[i], lower_bound_tasks[i], alpha=0.2, color=colors(i))

    # Vertical lines to separate tasks
    # for i in range(1, len(l_accuracies[0][0])):
    #     n_epochs_task = epochs[i-1] if isinstance(epochs, list) else epochs
    #     plt.axvline(x=i*n_epochs_task-1, color='grey',
    #                 linestyle='--', linewidth=0.5, zorder=0)

    # legend is the number of the task - Accuracy of the end of this task - accuracy at the end of all tasks
    legend = []
    for i in range(len(mean_accuracies[0])):
        index = sum(epochs[:i+1])-1 if isinstance(epochs,
                                                  list) else (i+1)*epochs-1
        end = sum(epochs)-1 if isinstance(epochs,
                                          list) else len(mean_accuracies)-1
        legend += [
            f"Task {i}: Epoch {index}: {mean_accuracies[index, i]:.2f}% - Epoch {end}: {mean_accuracies[-1, i]:.2f}%"]

    # Plot the average accuracy with end total accuracy
    average_accuracies = mean_accuracies.mean(dim=1)
    plt.plot(range(len(average_accuracies)),
             average_accuracies, color='black', linestyle='--', zorder=0, linewidth=0.75)
    # legend += ["Task change"]
    legend += [f"Average of tasks: {average_accuracies[-1]:.2f}%"]

    ### PLOT TRAINING ACCURACIES ###
    if training_accuracies is not None:
        training_accuracies = torch.tensor(training_accuracies).detach().cpu()
        mean_training_accuracies = training_accuracies.mean(dim=0)*100
        std_training_accuracies = training_accuracies.std(dim=0)*100

        # Plot the mean training accuracy
        for i in range(len(mean_training_accuracies[0])):
            plt.plot(range(len(mean_training_accuracies)),
                     mean_training_accuracies[:, i], linestyle='--', color=colors(i))

        # Fill between only accepts 1D arrays for error, we need to extract each std individually
        upper_bound_training_tasks, lower_bound_training_tasks = [], []
        for task in range(len(std_training_accuracies[0])):
            upper_bound_training_tasks.append(
                mean_training_accuracies[:, task] + std_training_accuracies[:, task])
            lower_bound_training_tasks.append(
                mean_training_accuracies[:, task] - std_training_accuracies[:, task])
        # Plot the std training accuracy
        for i in range(len(upper_bound_training_tasks)):
            plt.fill_between(range(len(mean_training_accuracies)),
                             upper_bound_training_tasks[i], lower_bound_training_tasks[i], alpha=0.2, color=colors(i))
    ### LEGEND ###
    plt.legend(
        legend,
        loc="center left",
        prop={'size': 6},
        frameon=False
    )
    ### SAVE ###
    plt.savefig(versionning(folder, title, ".pdf"), bbox_inches='tight')
    plt.savefig(versionning(folder, title, ".svg"), bbox_inches='tight')


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
    plt.legend(loc="lower right", frameon=False)

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
    plt.savefig(versionning(folder, title, ".pdf"), bbox_inches='tight')
    plt.savefig(versionning(folder, title, ".svg"), bbox_inches='tight')


def visualize_lr(parameters, lr, path, task=None, epoch=None):
    """ Plot a graph with the distribution in lambda values with respect to certain thresholds

    Args:
        parameters (dict): Parameters of the model
        lr (torch.Tensor): lr values
        threshold (int): Threshold to plot the distribution
        task (int): Task number
        epoch (int): Epoch number
    """

    params = [torch.zeros_like(param)
              for param in parameters if param.requires_grad]
    vector_to_parameters(lr, params)
    # figure with as many subplots as lambdas
    fig, ax = plt.subplots(len(params), 1, figsize=(5, 5*len(params)))
    for i, lr in enumerate(params):

        title = r'Learning rate ' + \
            f"[{'x'.join([str(s) for s in lr.shape][::-1])}]"

        bins = 100
        mini = lr.min().item()
        maxi = lr.max().item()
        hist = torch.histc(lr, bins=bins, min=mini,
                           max=maxi).detach().cpu()
        length = torch.prod(torch.tensor(lr.shape)).item()

        x = torch.linspace(mini, maxi, bins).detach().cpu()

        ax[i].bar(x,
                  hist * 100 / length,
                  width=2*(maxi-mini)/bins,
                  zorder=2,
                  color='purple')

        # write on the graph the maximum value and the minimum value
        ax[i].text(0.5, 0.95, f"Max: {maxi:.6f}",
                   fontsize=6, ha='center', va='center', transform=ax[i].transAxes)
        ax[i].text(0.5, 0.9, f"Min: {lr.min().item():.6f}",
                   fontsize=6, ha='center', va='center', transform=ax[i].transAxes)

        ax[i].set_xlabel(r'Learning rate [-]')
        ax[i].set_ylabel(r'Histogram of learning rate [%]')
        # font of xaxis label is 6
        ax[i].tick_params(axis='x', labelsize=6)
        ax[i].tick_params(which='both', width=1)
        ax[i].tick_params(which='major', length=6)
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].set_ylim(0, 100)
        ax[i].set_title(title, fontsize=8)

    os.makedirs(path, exist_ok=True)
    fig.savefig(versionning(path, f"lr-task{task}-epoch{epoch}" if epoch is not None else "lr",
                ".pdf"), bbox_inches='tight')
    plt.close()


def visualize_grad(parameters, grad, path, task=None, epoch=None):
    """ Plot a graph with the distribution in lambda values

    Args:
        parameters (dict): Parameters of the model
        grad (torch.Tensor): grad values
        path (str): Path to save the graph
        task (int): Task number
        epoch (int): Epoch number
    """

    params = [torch.zeros_like(param)
              for param in parameters if param.requires_grad]
    # if number of dim is greater than one
    if isinstance(grad, torch.Tensor):
        vector_to_parameters(grad, params)
    else:
        params = grad
    # figure with as many subplots as lambdas
    fig, ax = plt.subplots(len(params), 1, figsize=(5, 5*len(params)))
    for i, grad in enumerate(params):

        title = 'Gradient ' + \
            f"[{'x'.join([str(s) for s in grad.shape][::-1])}]"

        bins = 50
        length = torch.prod(torch.tensor(grad.shape)).item()
        hist = torch.histc(grad, bins=bins, min=-grad.max(),
                           max=grad.max()).detach().cpu()
        x = torch.linspace(-grad.max(), grad.max(), bins).detach().cpu()
        upper = grad.max().item()

        ax[i].bar(x,
                  hist * 100 / length,
                  width=2*upper/bins,
                  zorder=2,
                  color='purple')

        # write on the graph the maximum value and the minimum value
        ax[i].text(0.5, 0.95, f"Max: {grad.max():.6f}",
                   fontsize=6, ha='center', va='center', transform=ax[i].transAxes)
        ax[i].text(0.5, 0.9, f"Min: {grad.min():.6f}",
                   fontsize=6, ha='center', va='center', transform=ax[i].transAxes)

        ax[i].set_xlabel(r'Gradient [-]')
        ax[i].set_ylabel(r'Histogram of gradient [%]')
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        # font of xaxis label is 6
        ax[i].tick_params(axis='x', labelsize=6)
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
    if isinstance(lambda_, torch.Tensor):
        vector_to_parameters(lambda_, params)
    else:
        params = lambda_

    # figure with as many subplots as lambdas
    fig, ax = plt.subplots(len(params), 1, figsize=(5, 5*len(params)))
    for i, lbda in enumerate(params):

        title = r"$\lambda$" + \
            f"[{'x'.join([str(s) for s in lbda.shape][::-1])}]"

        bins = 100
        hist = torch.histc(lbda, bins=bins, min=-threshold,
                           max=threshold).detach().cpu()
        # Save the histogram to the computer
        histpath = path + f"/histograms/"
        os.makedirs(histpath, exist_ok=True)
        torch.save(hist, versionning(
            histpath, f"histogram-task{task}-epoch{epoch}-h{i}" if epoch is not None else f"histogram-h{i}", ".pt"))

        length = torch.prod(torch.tensor(lbda.shape)).item()
        # plot the histogram
        ax[i].bar(torch.linspace(-threshold, threshold, bins).detach().cpu(),
                  hist * 100 / length,
                  width=2*threshold/bins,
                  zorder=2)
        ax[i].set_xlabel('$\lambda$ [-]')
        ax[i].set_ylabel('Histogram of $\lambda$ [%]')
        ax[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[i].tick_params(which='both', width=1)
        ax[i].tick_params(which='major', length=6)
        ax[i].set_title(title, fontsize=8)
        ax[i].set_ylim(0, 50)

        textsize = 6
        transform = ax[i].transAxes
        # ax[i].text(0.5, 0.95, f"$\lambda$ values above {threshold}: {(lbda > threshold).sum() * 100 / length:.2f}%",
        #            fontsize=textsize, ha='center', va='center', transform=transform)
        # ax[i].text(0.5, 0.9, f"$\lambda$ values above 2: {((lbda > 2) & (lbda < threshold)).sum() * 100 / length:.2f}%",
        #            fontsize=textsize, ha='center', va='center', transform=transform)
        # ax[i].text(0.5, 0.85, f"$\lambda$  values below -2: {((lbda < -2) & (lbda > -threshold)).sum() * 100 / length:.2f}%",
        #            fontsize=textsize, ha='center', va='center', transform=transform)
        # ax[i].text(0.5, 0.8, f"$\lambda$ values below -{threshold}: {(lbda < -threshold).sum() * 100 / length:.2f}%",
        #            fontsize=textsize, ha='center', va='center', transform=transform)
        # ax[i].text(0.5, 0.75, f"$\lambda$ values between -2 and 2: {((lbda < 2) & (lbda > -2)).sum() * 100 / length:.2f}%",
        #            fontsize=textsize, ha='center', va='center', transform=transform)
        # print text with the mean value of lambda and the std
        ax[i].text(0.5, 0.95, f"Mean (abs): {torch.abs(lbda).mean().item():.6f}",
                   fontsize=textsize, ha='center', va='center', transform=transform)
        ax[i].text(0.5, 0.9, f"Std (abs): {torch.abs(lbda).std().item():.6f}",
                   fontsize=textsize, ha='center', va='center', transform=transform)

    os.makedirs(path, exist_ok=True)
    fig.savefig(versionning(path, f"lambda-task{task}-epoch{epoch}" if epoch is not None else "lambda",
                ".pdf"), bbox_inches='tight')
    fig.savefig(versionning(path, f"lambda-task{task}-epoch{epoch}" if epoch is not None else "lambda",
                ".svg"), bbox_inches='tight')
    plt.close()


def visualize_certainty(predictions, labels, path, task=None, epoch=None, log=True):
    """ Visualize the certainty of the model with respected to correct and incorrect predictions

    Args:
        predictions (torch.Tensor): Output of the layers of the model
        labels (torch.Tensor): Labels of the data
        path (str): Path to save the graph
        task (int): Task number
        epoch (int): Epoch number
        log (bool): If the predictions are in log space
    """
    if log:
        predictions = torch.exp(predictions)

    predicted = torch.argmax(torch.mean(predictions, dim=0), dim=1) == labels

    ### ALEATORIC UNCERTAINTY ###
    aleatoric = - torch.sum(torch.mean(predictions *
                            torch.log(predictions + 1e-8), dim=0), dim=-1)
    # We want to plot the histogram of aleatoric uncertainty for correct and incorrect predictions
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins = 50
    minimum = torch.min(aleatoric).item()
    maximum = torch.max(aleatoric).item()
    correct_hist = torch.histc(
        aleatoric[predicted], bins=bins, min=minimum, max=maximum).detach().cpu()

    incorrect_hist = torch.histc(
        aleatoric[~predicted], bins=bins, min=minimum, max=maximum).detach().cpu()

    # Plot as bars
    x = torch.linspace(minimum, maximum, bins).detach().cpu()
    plt.bar(x, correct_hist * 100 / len(aleatoric), width=3/bins,
            alpha=0.5, label="Correct predictions", color='blue')
    plt.bar(x, incorrect_hist * 100 /
            len(aleatoric), width=3/bins, alpha=0.5, label="Incorrect predictions", color='red')

    ax.set_xlabel('Aleatoric Uncertainty [-]')
    ax.set_ylabel('Histogram [%]')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    ax.set_ylim(0, 20)
    # save output
    fig.savefig(versionning(
        path, f"alea-certainty-task{task}-epoch{epoch}" if epoch is not None else "alea-certainty"),  bbox_inches='tight')
    plt.close()

    ### EPISTEMIC UNCERTAINTY ###
    # Epistemic uncertainty is the variance of the predictions
    mean_predictions = torch.mean(predictions, dim=0)
    predictive = - torch.sum(mean_predictions *
                             torch.log(mean_predictions + 1e-8), dim=-1)

    epistemic = predictive - aleatoric
    # We want to plot the histogram of aleatoric uncertainty for correct and incorrect predictions
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins = 50
    minimum = torch.min(epistemic).item()
    maximum = torch.max(epistemic).item()

    correct_hist = torch.histc(
        epistemic[predicted], bins=bins, min=minimum, max=maximum).detach().cpu()
    incorrect_hist = torch.histc(
        epistemic[~predicted], bins=bins, min=minimum, max=maximum).detach().cpu()
    x = torch.linspace(minimum, maximum, bins).detach().cpu()
    plt.bar(x,
            correct_hist * 100 / len(epistemic),
            width=maximum/bins,
            alpha=0.5,
            label="Correct predictions",
            color='blue')
    plt.bar(x,
            incorrect_hist * 100 / len(epistemic),
            width=maximum/bins,
            alpha=0.5,
            label="Incorrect predictions",
            color='red')
    ax.set_xlabel('Epistemic Uncertainty [-]')
    ax.set_ylabel('Histogram [%]')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    ax.set_ylim(0, 20)
    # save output
    fig.savefig(versionning(
        path, f"epi-certainty-task{task}-epoch{epoch}" if epoch is not None else "epi-certainty"),  bbox_inches='tight')
    plt.close()
