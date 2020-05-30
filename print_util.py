class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    BRIGHT_GREEN = '\033[92m'
    GREEN = '\033[32m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def log_acc_loss_header(color=''):
    print(color + 'Epoch'.ljust(11) + 'Iteration'.ljust(15) + 'Time'.ljust(10) + 'Loss'.ljust(10) + 'Accuracy'.ljust(14) + Color.END)

def log_acc_loss(epoch, nb_epoch, iteration, nb_iteration, time, score, loss, color='', persistent=True):
    print('\r' + color +
          '[{0}/{1}]'.format(epoch + 1, nb_epoch).ljust(11) +
          '[{0}/{1}]'.format(iteration + 1, nb_iteration).ljust(15) +
          '{0:.0f}s'.format(time).ljust(10) +
          '{0:.4f}'.format(loss).ljust(10) +
          (('.' * (round(time*1.5)%5)) if score is None else ('{0:.4f}'.format(score))).ljust(14) +
          Color.END,
          end='\n' if persistent else '')
