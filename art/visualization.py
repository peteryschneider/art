import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def plot_loss(train_losses, val_losses, save_dir):
    plt.figure()
    plt.plot(train_losses['action1'], label='Train')
    plt.plot(val_losses['action1'], label='Validation')
    plt.title('Loss: Action 1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_action1.png'))

    plt.figure()
    plt.plot(train_losses['action2'], label='Train')
    plt.plot(val_losses['action2'], label='Validation')
    plt.title('Loss: Action 2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_action2.png'))

    plt.figure()
    plt.plot(train_losses['state'], label='Train')
    plt.plot(val_losses['state'], label='Validation')
    plt.title('Loss: States')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_state.png'))

    plt.figure()
    plt.plot(train_losses['total'], label='Train')
    plt.plot(val_losses['total'], label='Validation')
    plt.title('Loss: Total')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_total.png'))