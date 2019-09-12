"""
This code is based on the QM9 dataset from the pythorch-geometric package,
see: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9
"""

import os
from six.moves import urllib
import torch
from os import makedirs
import tarfile
import numpy as np
import scipy.spatial.distance as dist
url = 'http://www.roemisch-drei.de/qm9.tar.gz'
raw_dir = os.path.join(os.getcwd(), 'data', 'QM9')
import pickle


def extract_tar(path, folder, mode='r:gz', log=True):
    r"""Extracts a tar archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        mode (string, optional): The compression mode. (default: :obj:`"r:gz"`)
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if log:
        print('Downloading', url)

    makedirs(folder)

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    path = os.path.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def download():
    file_path = download_url(url, raw_dir)
    extract_tar(file_path, raw_dir, mode='r')
    os.unlink(file_path)


def process():
    np.random.seed(seed=123)

    print('Processing data')
    raw_data_list = torch.load(os.path.join(raw_dir, 'qm9.pt'))
    dataset = []
    jj = 0
    for d in raw_data_list:
        if (jj % 1000) == 0:
            print(jj, '/', len(raw_data_list))
        jj = jj+1
        # get data
        x = d['x'].data.numpy()
        pos = d['pos'].data.numpy()
        y = d['y'].data.numpy()
        edge_index = d['edge_index'].data.numpy()
        edge_attr = d['edge_attr'].data.numpy()
        n = x.shape[0]
        # construct adjacency and distance matrices
        affinity = np.zeros((n,n))
        edge_features = np.zeros((n,n,4))
        for ii in range(edge_index.shape[1]):
            affinity[edge_index[0,ii],edge_index[1,ii]]=1
            edge_features[edge_index[0,ii],edge_index[1,ii]] = edge_attr[ii]
        distance_mat = dist.squareform(dist.pdist(pos))
        original_features = {'pos':pos,'edge_index':edge_index,'edge_attr':edge_attr,'pos':pos}
        usable_features = {'affinity':affinity,'distance_mat':distance_mat,'x':x,'edge_features':edge_features}

        # put in dictionary
        cdata = {'original_features':original_features,'usable_features':usable_features,'y':y}
        # add to list
        dataset.append(cdata)
    # shuffle
    print('Wrapping up')
    np.random.shuffle(dataset)

    tenpercent = int(len(dataset) * 0.1)

    test_dataset = dataset[:tenpercent]
    val_dataset = dataset[tenpercent:2 * tenpercent]
    train_dataset = dataset[2 * tenpercent:]

    pickle.dump(train_dataset,open(os.path.join(raw_dir, 'QM9_train.p'), 'wb'))
    pickle.dump(val_dataset, open(os.path.join(raw_dir, 'QM9_val.p'), 'wb'))
    pickle.dump(test_dataset, open(os.path.join(raw_dir, 'QM9_test.p'), 'wb'))


def main():
    download()
    process()


if __name__ == '__main__':
    main()

