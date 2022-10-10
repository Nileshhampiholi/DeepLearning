import os
import wget
import zipfile

# Download Annotations
url = r"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

wget.download(url)

# Extract Files
file_name = r"annotations_trainval2017.zip"
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall()

# delete zip file
os.remove(file_name)