import matplotlib.pyplot as plt
import numpy as np
from decentralized_SGD_logistic import DecentralizedSGDLogistic


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

def plot_losses_with_std(losses, iterations_indices, optimum_loss=0, title="My_nice_plot",
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
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    x = iterations_indices

    plt.figure(figsize=figsize)

    if len(losses) > 1:
        # If more than one topology
        for i in range(len(losses)):
            mean = np.vstack(losses[i]).mean(axis=0)
            std = np.vstack(losses[i]).std(axis=0)
            y = mean - optimum_loss

            if labels:
                plt.plot(x, y, label=labels[i], color=colors[i])
            else:
                plt.plot(x, y[i], color=colors[i])
            plt.fill_between(x, y-std, y+std, facecolor=colors[i], alpha=0.4)
    else:
        mean = np.vstack(losses[i]).mean(axis=0)
        std = np.vstack(losses[i]).std(axis=0)
        y = mean - optimum_loss

        if labels:
            plt.plot(x, y, label=labels, color=colors[0])
        else:
            plt.plot(x, y, color=colors[0])
        plt.fill_between(x, y-std, y+std, facecolor=colors[0], alpha=0.4)

    plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if labels:
        plt.legend(loc="best")

    if save_as_pdf:
        plt.savefig("plots/" + pdf_name + ".pdf", bbox_inches='tight', pad_inches=0)

    plt.show()

def load_csv_data(data_path):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    return yb, input_data

def clean(input_data, mean=False):

    #Replace -999 by most frequent value of column
    for i in range(input_data.shape[1]):
        current_col = input_data[:, i]

        if -999.0 in current_col:
            indices_to_change = (current_col == -999.0)
            if mean:
                curr_mean = np.mean(current_col[~indices_to_change])
                current_col[indices_to_change] = curr_mean
            else:
                (values,counts) = np.unique(current_col[~indices_to_change], return_counts=True)
                ind=np.argmax(counts)
                current_col[indices_to_change] = values[ind] if len(values) > 0 else 0

    return input_data

def standardize(x):
    """Standardize the given data"""
    means = x.mean(0)
    stds = x.std(0)
    return (x - means)/stds

def load_data():
    y, A = load_csv_data('train.csv')
    A = standardize(clean(A, True))
    y = 1 *(y > 0.0)
    return y, A

def run_logistic(A, y, param, logging=False):
    m = DecentralizedSGDLogistic(**param)
    list_losses = m.fit(A, y, logging=logging)
    if logging:
        print()
        print('Final score: {0:.4f}'.format(m.score(A, y)))
    return list_losses

def run_logistic_n_times(A, y, params, n):
    all_losses = []
    for i in range(n):
        print('Decentralized optimization, run number', i + 1, '\n')
        params['data_distribution_random_seed']= i + 1
        params['random_seed']= i + 1
        all_losses.append(run_logistic(A, y, params, logging=True))
    return all_losses
