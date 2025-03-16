import kagglehub

def download_dataset_from_kaggle(dataset_name):
    return kagglehub.dataset_download(dataset_name)