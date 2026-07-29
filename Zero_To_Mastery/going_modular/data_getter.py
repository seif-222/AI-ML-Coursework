import os
from os import mkdir
from pathlib import Path
import zipfile
import requests

def data_getter (data_path ,
                 image_path,
                 link,
                 remove_source=True):
    """getting Data"""
    data_path_fn = Path(data_path)
    image_path_fn = data_path_fn / image_path
    full_zip_name = str(image_path_fn)+".zip"

    if image_path_fn.is_dir():
        print("The file there ,  downloading skipped")
    else:
        print("The file is downloading..... ")
        image_path_fn.mkdir(parents=True , exist_ok=True)
        with open(full_zip_name, 'wb') as f:
            request = requests.get(link)
            f.write(request.content)
        with zipfile.ZipFile(full_zip_name) as zip_ref:
            print("unzipping .....")
            zip_ref.extractall(image_path_fn)
            print("done")
        if remove_source:
            os.remove(full_zip_name)
            print("removed source folder(zip file)")


