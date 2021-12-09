import pickle
import os
import sys
import matplotlib.pyplot as plt

def plot_save(name,x,save_path):
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title(name)
    plt.plot(range(len(x)), x, color='k')
    plt.ylabel('name')
    plt.xlabel('epochs')
    plt.savefig(save_path)

def plot(name,x):
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title(name)
    plt.plot(range(len(x)), x, color='k')
    plt.ylabel('name')
    plt.xlabel('epochs')


def plot_all(root='../save/'):
    all_path = ['cifar/iid','cifar/no_iid','mnist/iid','mnist/no_iid']
    # all_path = os.listdir(root)
    all_path = [os.path.join(root,p) for p in all_path]
    for p in all_path:
        for file in os.listdir(p):
            if file.endswith('pkl'):
                data = pickle.load(open(os.path.join(p,file),'rb'))
                save_root = os.path.join(p,file.split('.')[0])
                os.makedirs(save_root,exist_ok=True)
                if 'col_performance' in file:
                    plt.figure()
                    plt.title('acc')
                    plt.ylabel('acc')
                    plt.xlabel('epochs')
                    for i in data.keys():
                        x = data[i]
                        plt.plot(range(len(x)), x)
                    plt.savefig(os.path.join(save_root,'acc.png'))
                else:
                    for k in data[0].keys():
                        plt.figure()
                        plt.title(k)
                        plt.ylabel(k)
                        plt.xlabel('epochs')
                        for i in range(len(data)):
                            x = data[i][k]
                            plt.plot(range(len(x)), x)
                        plt.savefig(os.path.join(save_root,k+'.png'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = '../tensorflow_result/'
    plot_all(root)