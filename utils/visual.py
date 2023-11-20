import os
import matplotlib.pyplot as plt
import datetime


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


def visualize_sequential(title, t_accuracy, folder):
    plt.figure()
    plt.plot(range(1, len(t_accuracy)+1), t_accuracy, zorder=3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend([f"Task {i+1}" for i in range(len(t_accuracy))])
    n_epochs_task = len(t_accuracy) // len(t_accuracy[0])
    # Vertical lines to separate tasks
    for i in range(1, len(t_accuracy[0])):
        plt.axvline(x=i*n_epochs_task, color='k',
                    linestyle='--', linewidth=0.5)
    os.makedirs(folder, exist_ok=True)
    versionned = versionning(folder, title)
    plt.savefig(versionned, bbox_inches='tight')
