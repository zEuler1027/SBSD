from qm9.data.prepare import download_dataset_qm9


# Download the QM9 dataset and prepare it for training.
if __name__ == '__main__':
    download_dataset_qm9(datadir='./data', dataname='qm9')
