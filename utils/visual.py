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
    plt.plot(range(1, len(t_accuracy)+1), t_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['MNIST', 'Fashion-MNIST'])
    os.makedirs(folder, exist_ok=True)
    versionned = versionning(folder, title)
    plt.savefig(versionned, bbox_inches='tight')
