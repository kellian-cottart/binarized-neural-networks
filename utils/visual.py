import os
import matplotlib.pyplot as plt
import datetime
import torch
from matplotlib.ticker import AutoMinorLocator
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from optimizers import *

# plt.switch_backend('agg')


def graphs(main_folder, net_trainer, task, n_tasks, epoch, predictions=None, labels=None, ood_predictions=None, ood_labels=None):

    # print predicted and associated certainty
    visualize_certainty_task(
        predictions=predictions,
        labels=labels,
        path=os.path.join(main_folder, "certainty"),
        task=task,
        epoch=epoch,
        log=True,
        ood_predictions=ood_predictions,
        ood_labels=ood_labels,
    )
    params = list(net_trainer.model.parameters())
    grad = [param.grad for param in params if param.grad is not None]
    if grad != []:
        visualize_grad(
            parameters=params,
            grad=grad,
            path=os.path.join(main_folder, "grad"),
            task=task+1,
            epoch=epoch+1,
        )
    visualize_lambda(
        parameters=params,
        lambda_=params if not isinstance(
            net_trainer.optimizer, BayesBiNN) else net_trainer.optimizer.state['lambda'],
        path=os.path.join(main_folder, "lambda"),
        threshold=None,
        task=task+1,
        epoch=epoch+1,
    )
    if hasattr(net_trainer.optimizer, 'state') and net_trainer.optimizer.state['lr'] != {}:
        visualize_lr(
            parameters=params,
            lr=net_trainer.optimizer.state['lr'],
            path=os.path.join(main_folder, "lr"),
            task=task+1,
            epoch=epoch+1,
        )


def versionning(folder, title, format=".pdf"):
    os.makedirs(folder, exist_ok=True)
    # YYYY-MM-DD-hh-mm-ss-title-version.format
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    # Set minor ticks
    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    ### COMPUTE MEAN AND STD ###
    # Transform the list of list of accuracies into a tensor of tensor of accuracies
    l_accuracies = l_accuracies.detach().cpu()
    mean_accuracies = l_accuracies.mean(dim=0)*100
    std_accuracies = l_accuracies.std(dim=0)*100
    ### PLOT ###
    # Create a gradient of len(l_accuracies[0]) colors corresponding to each task.
    if len(mean_accuracies[0]) > 2:
        colors = plt.get_cmap('viridis', len(mean_accuracies[0]))
    else:
        colors = plt.get_cmap('Spectral_r', len(mean_accuracies[0]))

    # Plot the maximum accuracy as a horizontal line
    plt.axhline(y=mean_accuracies.max(), color='grey',
                linestyle='--', linewidth=1, label='Top accuracy: {:.2f}%'.format(mean_accuracies.max()))
    # Plot the mean accuracy
    for i in range(len(mean_accuracies[0])):
        plt.plot(range(len(mean_accuracies)),
                 mean_accuracies[:, i], color=colors(i), alpha=0.8, label=f"Task {i+1}")
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
    # Plot the average accuracy with end total accuracy
    average_accuracies = mean_accuracies.mean(dim=1)
    # plot text
    if epochs is not None:
        # average accuracy of the tasks
        plt.text(0.8, 0.15, f"Average of tasks: {average_accuracies[-1]:.2f}%",
                 fontsize=9, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
        # Difference between first and last task at the last epoch
        if isinstance(epochs, int):
            difference = mean_accuracies[epochs-1, 0] - mean_accuracies[-1, -1]
        else:
            difference = mean_accuracies[epochs[0]-1,
                                         0] - mean_accuracies[epochs[-1]-1, -1]
        plt.text(0.8, 0.10, f"Vanishing Plasticity: {difference:.2f}%",
                 fontsize=9, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    ### PLOT TRAINING ACCURACIES ###
    if training_accuracies is not None:
        training_accuracies = training_accuracies.detach().cpu()
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
    ### LABELS ###
    plt.xlabel('Epochs [-]')
    plt.ylabel('Accuracies [%]')
    plt.ylim(0, 100)
    new_ticks = [str(i) for i in range(
        1, len(mean_accuracies)+1, len(mean_accuracies) // 20)] if len(mean_accuracies) >= 20 else [str(i) for i in range(1, len(mean_accuracies)+1)]
    range_ticks = torch.arange(0, len(mean_accuracies), len(mean_accuracies) // 20) if len(
        mean_accuracies) >= 20 else torch.arange(0, len(mean_accuracies))
    plt.xticks(range_ticks.cpu(), new_ticks)
    plt.xlim(0, len(mean_accuracies)-1)
    ### LEGEND ###
    plt.legend(
        loc="center right",
        prop={'size': 9},
        frameon=False,
        fancybox=True,
        framealpha=0.8,
    )
    ### SAVE ###
    plt.savefig(versionning(folder, title, ".pdf"), bbox_inches='tight')
    plt.savefig(versionning(folder, title, ".svg"), bbox_inches='tight')
    plt.close()


def get_mean_std_accuracies(l_accuracies, t_start, t_end):
    # Compute the number of epochs
    n_epochs = len(l_accuracies[0]) // len(l_accuracies[0][0])
    # l_accuracies is a vector of n network accuracies
    # l_accuracies[0] is the accuracy of the first network for each task
    l_accuracies = l_accuracies.detach().cpu()
    mean_acc = torch.mean(l_accuracies, dim=0)
    std_acc = l_accuracies.std(dim=0)
    # Get the last epochs at t_end
    final_epoch = t_end * n_epochs - 1
    mean_acc = mean_acc[final_epoch]
    std_acc = std_acc[final_epoch]
    # Retrieve only the tasks between t_start and t_end
    mean_acc = mean_acc[t_start-1:t_end+1] * 100
    std_acc = std_acc[t_start-1:t_end+1] * 100
    return mean_acc, std_acc


def visualize_task_frame(title, l_accuracies, folder, t_start, t_end):
    """ Visualize the accuracy of each task between t_start and t_end

    Args:
        title (str): title of the figure
        l_accuracies (list): list of list of accuracies for each task at each epoch
        folder (str): folder to save the figure
        t_start (int): start task
        t_end (int): end task (included)
    """
    mean_acc, std_acc = get_mean_std_accuracies(l_accuracies, t_start, t_end)
    plt.figure()
    # Scatter with line
    plt.plot(range(len(mean_acc)), mean_acc,
             zorder=3, marker='o', color='purple')
    # Fill between std
    plt.fill_between(range(len(mean_acc)), mean_acc-std_acc,
                     mean_acc+std_acc, alpha=0.3, zorder=2, color='purple')
    plt.legend(loc="lower right", frameon=False)

    # hline with top accuracy
    plt.axhline(y=mean_acc.max(), color='grey',
                linestyle='--', linewidth=1, label='Top accuracy: {:.2f}%'.format(mean_acc.max()))

    plt.xlim(t_start, t_end)
    plt.xlabel('Task [-]')
    plt.ylabel('Accuracy [%]')
    # xtickslabels from t_start to t_end as string
    plt.xticks(torch.arange(0, t_end+1-t_start).detach().cpu(),
               [str(i) for i in range(t_start, t_end+1)])
    plt.xlim(0, t_end-t_start)
    plt.ylim(60, 100)

    # increase the size of the ticks
    ax = plt.gca()
    ax.legend(loc="lower right", prop={'size': 12}, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.tick_params(axis='y', which='minor', length=2)
    ### SAVE ###
    output = versionning(folder, title, ".pdf")
    plt.savefig(output, bbox_inches='tight')
    plt.savefig(output, bbox_inches='tight')
    plt.close()


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
    if isinstance(lr, torch.Tensor):
        vector_to_parameters(lr, params)
    else:
        params = lr
    fig, ax = plt.subplots(len(params), 1, figsize=(5, 5*len(params)))
    for i, lr in enumerate(params):
        current_ax = ax[i] if len(params) > 1 else ax
        title = r'Learning rate ' + \
            f"[{'x'.join([str(s) for s in lr.shape][::-1])}]"

        bins = 100
        mini = lr.min().item()
        maxi = lr.max().item()
        hist = torch.histc(lr, bins=bins, min=mini,
                           max=maxi).detach().cpu()
        length = torch.prod(torch.tensor(lr.shape)).item()

        x = torch.linspace(mini, maxi, bins).detach().cpu()

        current_ax.bar(x,
                       hist * 100 / length,
                       width=2*(maxi-mini)/bins,
                       zorder=2,
                       color='purple')

        # write on the graph the maximum value and the minimum value
        current_ax.text(0.5, 0.95, f"Max: {maxi:.6f}",
                        fontsize=6, ha='center', va='center', transform=current_ax.transAxes)
        current_ax.text(0.5, 0.9, f"Min: {lr.min().item():.6f}",
                        fontsize=6, ha='center', va='center', transform=current_ax.transAxes)

        current_ax.set_xlabel(r'Learning rate [-]')
        current_ax.set_ylabel(r'Histogram of learning rate [%]')
        # font of xaxis label is 6
        current_ax.tick_params(axis='x', labelsize=6)
        current_ax.tick_params(which='both', width=1)
        current_ax.tick_params(which='major', length=6)
        current_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        current_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        current_ax.set_ylim(0, 100)
        current_ax.set_title(title, fontsize=8)

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
        current_ax = ax[i] if len(params) > 1 else ax
        title = 'Gradient ' + \
            f"[{'x'.join([str(s) for s in grad.shape][::-1])}]"

        bins = 50
        length = torch.prod(torch.tensor(grad.shape)).item()
        hist = torch.histc(grad, bins=bins, min=-grad.max(),
                           max=grad.max()).detach().cpu()
        x = torch.linspace(-grad.max(), grad.max(), bins).detach().cpu()
        upper = grad.max().item()

        current_ax.bar(x,
                       hist * 100 / length,
                       width=2*upper/bins,
                       zorder=2,
                       color='purple')

        # write on the graph the maximum value and the minimum value
        current_ax.text(0.5, 0.95, f"Mean (abs): {torch.abs(grad).mean().item():.6f}",
                        fontsize=6, ha='center', va='center', transform=current_ax.transAxes)
        current_ax.text(0.5, 0.9, f"Std (abs): {torch.abs(grad).std().item():.6f}",
                        fontsize=6, ha='center', va='center', transform=current_ax.transAxes)

        current_ax.set_xlabel(r'Gradient [-]')
        current_ax.set_ylabel(r'Histogram of gradient [%]')
        current_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        current_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        # font of xaxis label is 6
        current_ax.tick_params(axis='x', labelsize=6)
        current_ax.tick_params(which='both', width=1)
        current_ax.tick_params(which='major', length=6)
        current_ax.set_ylim(0, 100)
        current_ax.set_title(title, fontsize=8)

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
        current_ax = ax[i] if len(params) > 1 else ax
        title = r"$\lambda$" + \
            f"[{'x'.join([str(s) for s in lbda.shape][::-1])}]"
        bins = 100
        threshold = torch.max(torch.abs(lbda)).item()
        hist = torch.histc(lbda, bins=bins, min=-threshold,
                           max=threshold).detach().cpu()
        # Save the histogram to the computer
        histpath = path + f"/histograms/"
        os.makedirs(histpath, exist_ok=True)
        torch.save(hist, versionning(
            histpath, f"histogram-task{task}-epoch{epoch}-h{i}" if epoch is not None else f"histogram-h{i}", ".pt"))

        length = torch.prod(torch.tensor(lbda.shape)).item()
        # plot the histogram
        current_ax.bar(torch.linspace(-threshold, threshold, bins).detach().cpu(),
                       hist * 100 / length,
                       width=2*threshold/bins,
                       zorder=2)
        current_ax.set_xlabel('$\lambda$ [-]')
        current_ax.set_ylabel('Histogram of $\lambda$ [%]')
        current_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        current_ax.tick_params(which='both', width=1)
        current_ax.tick_params(which='major', length=6)
        current_ax.set_title(title, fontsize=8)
        current_ax.set_ylim(0, 50)

        textsize = 6
        transform = current_ax.transAxes
        # print text with the mean value of lambda and the std
        current_ax.text(0.5, 0.95, f"Mean (abs): {torch.abs(lbda).mean().item():.6f}",
                        fontsize=textsize, ha='center', va='center', transform=transform)
        current_ax.text(0.5, 0.9, f"Std (abs): {torch.abs(lbda).std().item():.6f}",
                        fontsize=textsize, ha='center', va='center', transform=transform)

    os.makedirs(path, exist_ok=True)
    fig.savefig(versionning(path, f"lambda-task{task}-epoch{epoch}" if epoch is not None else "lambda",
                ".pdf"), bbox_inches='tight')
    fig.savefig(versionning(path, f"lambda-task{task}-epoch{epoch}" if epoch is not None else "lambda",
                ".svg"), bbox_inches='tight')
    plt.close()


def visualize_certainty_task(predictions, labels, path, task=None, epoch=None, log=True, ood_predictions=None, ood_labels=None):
    """ Visualize the certainty of the model with respected to correct and incorrect predictions

    Args:
        predictions (torch.Tensor): Output of the layers of the model
        labels (torch.Tensor): Labels of the data
        path (str): Path to save the graph
        task (int): Task number
        epoch (int): Epoch number
        log (bool): If the predictions are in log space
    """
    # Concatenate the predictions
    concat_predictions = torch.cat(predictions, dim=1)
    if log == True:
        concat_predictions = torch.exp(concat_predictions)
    concat_labels = torch.cat(labels, dim=0)

    aleatoric = torch.zeros(
        (concat_predictions.shape[2], concat_predictions.shape[1]))
    epistemic = torch.zeros(
        (concat_predictions.shape[2], concat_predictions.shape[1]))
    # Compute the uncertainty associated with each class
    for k in concat_labels.unique():
        alea_uncertainty, epi_uncertainty = compute_task_uncertainty(
            concat_predictions[:, :, k])
        aleatoric[k] = alea_uncertainty
        epistemic[k] = epi_uncertainty
    # vectors are (n_classes, n_elements)
    sum_aleatoric = torch.sum(aleatoric, dim=0)
    sum_epistemic = torch.sum(epistemic, dim=0)
    seen_indexes = torch.cat(
        [pred for pred in predictions[:task+1]], dim=1).shape[1]
    mean_predictions = torch.mean(concat_predictions, dim=0)
    # in the seen indexes, get the right predictions and the wrong predictions
    correct_predictions = torch.argmax(
        mean_predictions[:seen_indexes, :], dim=1) == concat_labels[:seen_indexes].to(mean_predictions.device)
    false_predictions = torch.argmax(
        mean_predictions[:seen_indexes, :], dim=1) != concat_labels[:seen_indexes].to(mean_predictions.device)
    aleatoric_unseen = sum_aleatoric[seen_indexes:]
    aleatoric_seen = sum_aleatoric[:seen_indexes]
    epistemic_seen = sum_epistemic[:seen_indexes]
    epistemic_unseen = sum_epistemic[seen_indexes:]

    graph_uncertainty(path=path,
                      title=f"alea-certainty-{epoch+1}-task{task+1}" if epoch is not None else "alea-certainty",
                      seen_uncertainty=aleatoric_seen,
                      unseen_uncertainty=aleatoric_unseen,
                      correct_indices=correct_predictions,
                      false_indices=false_predictions,
                      )
    graph_uncertainty(path=path,
                      title=f"epi-certainty-{epoch+1}-task{task+1}" if epoch is not None else "epi-certainty",
                      seen_uncertainty=epistemic_seen,
                      unseen_uncertainty=epistemic_unseen,
                      correct_indices=correct_predictions,
                      false_indices=false_predictions,
                      )
    if ood_predictions is not None:
        ood_concat_predictions = torch.cat(ood_predictions, dim=1)
        print(ood_concat_predictions.shape, concat_predictions.shape)
        if log == True:
            ood_concat_predictions = torch.exp(ood_concat_predictions)
        ood_concat_labels = torch.cat(ood_labels, dim=0)
        ood_aleatoric = torch.zeros(
            (ood_concat_predictions.shape[2], ood_concat_predictions.shape[1]))
        ood_epistemic = torch.zeros(
            (ood_concat_predictions.shape[2], ood_concat_predictions.shape[1]))
        # Compute the uncertainty associated with each class
        for k in ood_concat_labels.unique():
            ood_alea_uncertainty, ood_epi_uncertainty = compute_task_uncertainty(
                ood_concat_predictions[:, :, k])
            ood_aleatoric[k] = ood_alea_uncertainty
            ood_epistemic[k] = ood_epi_uncertainty
        # vectors are (n_classes, n_elements)
        ood_sum_aleatoric = torch.sum(ood_aleatoric, dim=0)
        ood_sum_epistemic = torch.sum(ood_epistemic, dim=0)
        graph_uncertainty(path=path,
                          title=f"ood-alea-certainty-{epoch+1}-task{task+1}" if epoch is not None else "ood-alea-certainty",
                          seen_uncertainty=aleatoric_seen,
                          unseen_uncertainty=ood_sum_aleatoric,
                          correct_indices=correct_predictions,
                          false_indices=false_predictions,
                          )
        graph_uncertainty(path=path,
                          title=f"ood-epi-certainty-{epoch+1}-task{task+1}" if epoch is not None else "ood-epi-certainty",
                          seen_uncertainty=sum_epistemic[:seen_indexes],
                          unseen_uncertainty=ood_sum_epistemic,
                          correct_indices=correct_predictions,
                          false_indices=false_predictions,
                          )

    # save pt file for the uncertainty
    torch.save(aleatoric, versionning(
        path, f"alea-certainty-{epoch+1}-task{task+1}" if epoch is not None else "alea-certainty", ".pt"))
    torch.save(epistemic, versionning(
        path, f"epi-certainty-{epoch+1}-task{task+1}" if epoch is not None else "epi-certainty", ".pt"))
    if ood_predictions is not None:
        torch.save(ood_aleatoric, versionning(
            path, f"ood-alea-certainty-{epoch+1}-task{task+1}" if epoch is not None else "ood-alea-certainty", ".pt"))
        torch.save(ood_epistemic, versionning(
            path, f"ood-epi-certainty-{epoch+1}-task{task+1}" if epoch is not None else "ood-epi-certainty", ".pt"))

    # # bar plot of the epistemic uncertainty per class
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.bar(torch.arange(0, epistemic.shape[0]).detach().cpu(),
    #        epistemic[:, 0].detach().cpu(), color='blue', label='Epistemic')
    # # stack bar aleatoric
    # ax.bar(torch.arange(0, aleatoric.shape[0]).detach().cpu(),
    #        aleatoric[:, 0].detach().cpu(), bottom=epistemic[:, 0].detach().cpu(), color='red', label='Aleatoric')
    # ax.set_xlabel('Class [-]')
    # ax.set_ylabel('Uncertainty [-]')
    # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.set_xticks(torch.arange(0, epistemic.shape[0]).detach().cpu())
    # ax.text(0.5, 0.95, f"Predicted class: {torch.argmax(mean_predictions[0]).item()}", fontsize=9,
    #         ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    # ax.text(0.5, 0.9, f"True class: {concat_labels[0].item()}", fontsize=9,
    #         ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    # ax.tick_params(which='both', width=1)
    # ax.tick_params(which='major', length=6)
    # ax.legend(frameon=False)
    # ax.set_ylim(0, 0.75)
    # # save output
    # fig.savefig(versionning(
    #     path, f"uncertaintyseen0-task{task+1}-epoch{epoch+1}" if epoch is not None else "uncertainty"),  bbox_inches='tight')
    # plt.close()
    # if len(epistemic_unseen) == 0:
    #     return
    # # bar plot of the epistemic uncertainty on the first unseen
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.bar(torch.arange(0, epistemic.shape[0]).detach().cpu(),
    #        epistemic[:, seen_indexes+1].detach().cpu(), color='blue', label='Epistemic')
    # # stack bar aleatoric
    # ax.bar(torch.arange(0, aleatoric.shape[0]).detach().cpu(),
    #        aleatoric[:, seen_indexes+1].detach().cpu(), bottom=epistemic[:, seen_indexes+1].detach().cpu(), color='red', label='Aleatoric')
    # ax.set_xlabel('Class [-]')
    # ax.set_ylabel('Uncertainty [-]')
    # ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.set_xticks(torch.arange(0, epistemic.shape[0]).detach().cpu())
    # ax.text(0.5, 0.95, f"Predicted class: {torch.argmax(mean_predictions[seen_indexes+1]).item()}", fontsize=9,
    #         ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    # ax.text(0.5, 0.9, f"True class: {concat_labels[seen_indexes+1].item()}", fontsize=9,
    #         ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    # ax.tick_params(which='both', width=1)
    # ax.tick_params(which='major', length=6)
    # ax.legend(frameon=False)
    # ax.set_ylim(0, 0.75)
    # # save output
    # fig.savefig(versionning(
    #     path, f"uncertaintyunseen0-task{task+1}-epoch{epoch+1}" if epoch is not None else "uncertainty"),  bbox_inches='tight')
    # plt.close()
    # We want to plot the ROC curve of the model between seen and unseen distributions
    if len(epistemic_unseen) == 0:
        return
    roc_auc(path=path,
            title=f"roc-auc-epistemic-task{task+1}-epoch{epoch+1}" if epoch is not None else "roc-auc",
            epistemic_seen=epistemic_seen,
            epistemic_unseen=epistemic_unseen,
            )
    if ood_predictions is not None:
        roc_auc(path=path,
                title=f"roc-auc-ood-epistemic-task{task+1}-epoch{epoch+1}" if epoch is not None else "roc-auc",
                epistemic_seen=epistemic_seen,
                epistemic_unseen=ood_sum_epistemic,
                )


def roc_auc(path, title, epistemic_seen, epistemic_unseen):
    threshold = torch.linspace(0, 1, 100).cpu()
    fpr = torch.zeros_like(threshold)
    tpr = torch.zeros_like(threshold)
    for i, t in enumerate(threshold):
        fpr[i] = torch.sum(epistemic_seen > t).item() / len(epistemic_seen)
        tpr[i] = torch.sum(epistemic_unseen > t).item() / len(epistemic_unseen)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.text(0.5, 0.95, f"AUC: {-torch.trapz(tpr, fpr).item():.2f}",
            fontsize=9, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    ax.plot(fpr, tpr, label="ROC curve", color='purple')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set_xlabel('Unseen Rate [-]')
    ax.set_ylabel('Seen Rate [-]')
    ax.legend()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    # save output
    fig.savefig(versionning(path, title),  bbox_inches='tight')
    plt.close()


def graph_uncertainty(path, title, seen_uncertainty, unseen_uncertainty, correct_indices, false_indices):
    bins = 50
    if len(unseen_uncertainty) == 0:
        minimum = torch.min(seen_uncertainty).item()
        maximum = torch.max(seen_uncertainty).item()
    else:
        minimum = torch.min(seen_uncertainty).item()
        maximum = torch.max(unseen_uncertainty).item()
    correct_hist = torch.histc(
        seen_uncertainty[correct_indices], bins=bins, min=minimum, max=maximum).detach().cpu()
    incorrect_hist = torch.histc(
        seen_uncertainty[false_indices], bins=bins, min=minimum, max=maximum).detach().cpu()
    unseen_hist = torch.histc(
        unseen_uncertainty, bins=bins, min=minimum, max=maximum).detach().cpu()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    width = (maximum - minimum) / bins
    ax.bar(torch.linspace(minimum, maximum, bins).detach().cpu(),
           correct_hist * 100 / len(seen_uncertainty), width=width, alpha=0.5, label="Correct predictions", color='blue')
    ax.bar(torch.linspace(minimum, maximum, bins).detach().cpu(),
           incorrect_hist * 100 / len(seen_uncertainty), width=width, alpha=0.5, label="Incorrect predictions", color='orange')
    ax.bar(torch.linspace(minimum, maximum, bins).detach().cpu(),
           unseen_hist * 100 / len(unseen_uncertainty), width=width, alpha=0.5, label="Unseen predictions", color='red')
    ax.set_xlabel('Uncertainty [-]')
    ax.set_ylabel('Histogram [%]')
    ax.legend(frameon=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which='both', width=1)
    ax.tick_params(which='major', length=6)
    ax.set_ylim(0, 50)
    # save output
    fig.savefig(versionning(path, title), bbox_inches='tight')
    plt.close()


def compute_task_uncertainty(predictions_k):
    """ Compute the aleatoric and epistemic uncertainty for a given class k

    Args:
        k (int): Class number
        predictions (torch.Tensor): Predictions of the model (shape (n_samples, n_elements, n_classes))
    Returns:
        aleatoric_uncertainty (torch.Tensor): Aleatoric uncertainty (shape (n_elements))
        epistemic_uncertainty (torch.Tensor): Epistemic uncertainty (shape (n_elements))
    """
    # TOTAL UNCERTAINTY ###
    mean_predictions = torch.mean(predictions_k, dim=0)
    total_uncertainty = -mean_predictions * torch.log2(mean_predictions)
    epistemic_uncertainty = torch.zeros(
        (predictions_k.shape[0], predictions_k.shape[1]))
    for i in range(len(predictions_k)):
        # KL divergence between predictions and mean predictions
        epistemic_uncertainty[i] = predictions_k[i] * \
            (torch.log2(predictions_k[i] + 1e-8) -
             torch.log2(mean_predictions + 1e-8))
    epistemic_uncertainty = torch.mean(epistemic_uncertainty, dim=0)
    aleatoric_uncertainty = total_uncertainty - epistemic_uncertainty
    return aleatoric_uncertainty, epistemic_uncertainty
