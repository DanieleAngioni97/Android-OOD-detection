import gdown, zipfile

if __name__ == '__main__':
    data_id = "1HZ1JilWfq7Yz259tjb96ztcsPyGtn254"
    data_folder = "data/"

    gdown.download(id=data_id, output="data.zip")
    zip_file_path = "data.zip"

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents to the specified directory
        zip_ref.extractall(data_folder)