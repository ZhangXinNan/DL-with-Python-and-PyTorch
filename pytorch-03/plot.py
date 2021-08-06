
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losses, eval_losses, save_path):
    plt.clf()   # clear figure
    plt.title('train-val-loss')
    plt.plot(np.arange(len(losses)), losses)
    plt.plot(np.arange(len(eval_losses)), eval_losses)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.legend(['Train Loss'], loc='upper right')
    plt.savefig(save_path)


def plot_acc(acces, eval_acces, save_path):
    plt.clf()   # clear figure
    plt.title('train-val-acc')
    plt.plot(np.arange(len(acces)), acces)
    plt.plot(np.arange(len(eval_acces)), eval_acces)
    plt.legend(['Train acc', 'Test acc'], loc='upper right')
    plt.legend(['Train acc'], loc='upper right')
    plt.savefig(save_path)


