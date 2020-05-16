import matplotlib.pyplot as plt
import numpy as np


def plot_losses(losses, iterations_indices, optimum_loss=0, title="My_nice_plot",
                xlabel="Iteration", ylabel="Error", labels=None, yscale="log", ylim=None,
                figsize=(10, 5), save_as_pdf=False, pdf_name="my_nice_plot_in_PDF"):
    """Plot the different losses with values shifted by the value of the optimal loss.

    :param losses: 2D Array containing the losses for each topology
    :param iterations_indices: 1D Array containing the indices (in number of iteration) of the given losses
    :param optimum_loss: Value of the optimal loss
    :param title: Title of the plot
    :param xlabel: Label of the x axis
    :param ylabel: Label of the y axis
    :param labels: Labels of the different topologies, e.g., labels=["fully connected", "ring", "torus"]
    :param yscale: Scale of the y axis, e.g., "linear", "log",...
    :param ylim: Limits of the y axis, e.g., ylim=(0.001, 1)
    :param figsize: Size of the plot
    :param save_as_pdf: Saves the plot as pdf if True
    :param pdf_name: If the plot is saved as pdf, saves with this name
    """
    y = np.array(losses).squeeze()
    y -= optimum_loss
    x = iterations_indices

    plt.figure(figsize=figsize)

    if len(y.shape) > 1:
        # If more than one topology
        for i in range(y.shape[0]):
            if labels:
                plt.plot(x, y[i], label=labels[i])
            else:
                plt.plot(x, y[i])
    else:
        if labels:
            plt.plot(x, y, label=labels)
        else:
            plt.plot(x, y)

    plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if labels:
        plt.legend(loc="best")

    if save_as_pdf:
        plt.savefig("plots/" + pdf_name + ".pdf", bbox_inches='tight')

    plt.show()