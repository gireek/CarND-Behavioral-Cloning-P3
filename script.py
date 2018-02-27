import os
from urllib.request import urlretrieve

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')
download('https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip', 'download.zip')
import zipfile
zip_ref = zipfile.ZipFile('download.zip', 'r')
zip_ref.extractall(".")
zip_ref.close()