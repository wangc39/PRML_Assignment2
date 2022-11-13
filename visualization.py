import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, y, values=[0], idxs=[0], interval=None, output_dir='visualization/', labels=['train', 'test'],
                xlabel='Number of Eigenvectors', ylabel='Accumulative Explained Variance Ratio',
                xlim=None, ylim=None):
    '''
        y: [[]] y data, two-dim list
        x: [[]]
        idxs: [[]]
        values: [[]]
    '''

    colors = ['b', 'r', 'c', 'y', 'b', 'k', ]
    
    fig = plt.figure(figsize=(16, 8))

    if y.ndim == 1:
        y = np.array([y])

    if x is None:
        x = np.array([np.arange(len(y[i])) for i in range(y.ndim)])

    # print(x)
    if interval is not None:
        for i in range(y.ndim):
            y[i] = y[i][::interval] 
            x[i] = np.arange(0, len(y), interval)

    # if isinstance(values, (float, int)):
    #     values = np.array([[values]]*y.ndim)
    # elif isinstance(values, list):
    #     values = np.array([values]*y.ndim)
    # else:
    #     raise Exception("Unknow type of values", values)
    
    # if isinstance(idxs, (float, int)):
    #     idxs = np.array([[idxs]]*y.ndim)
    # elif isinstance(idxs, list):
    #     idxs = np.array([idxs]*y.ndim)
    # else:
    #     raise Exception("Unknow type of idxs", idxs)


    # plt.plot(x, y, color='r', linestyle='-', marker='o', markersize=5, markerfacecolor='k')

    for i in range(y.ndim):
        print(x[i], y[i])
        plt.plot(x[i], y[i], '{}o-'.format(colors[i]), label=labels[i])


        for k, v in zip(idxs[i], values[i]):
            plt.plot([k]*2, [0, v], 'k--')
            plt.plot([0, k], [v]*2, 'k--')
            print(k, v)
            plt.text(k, v, '({},{:.2f})'.format(k, v), fontdict={'fontsize': 15})

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir)
    plt.show()


def plot_gallery(images, titles, h, w, n_row=3, n_col=4, save_path=None):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    
    if save_path: plt.savefig(os.path.join(save_path, 'gallery.jpg'))
 
 
# plot the result of the prediction on a portion of the test set
 
def get_title(y_pred, y_test, target_names, i):
    pred_name = target_names[int(y_pred[i]) - 1]
    true_name = target_names[int(y_test[i]) - 1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
