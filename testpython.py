'''
# First python file
print("Hello_World")
print("Hello_Greece")
print("Hello_Mytilini")
print("Hello_Malesiada")
print("Hello_public_address")
print("Kalo mina with ngrok!")
print("Kala Xristougenna")
print("Kali Xronia")
print("new jenkins 3")
'''
import kaggle 
import zipfile
from numpy import loadtxt
import numpy as np
import pandas as pd
import os


# Authentication
# Perform the authentication in the notebook directly by using the OS environment variables
# import os
# os.environ['KAGGLE_USERNAME'] = "<your-kaggle-username>"
# os.environ['KAGGLE_KEY'] = "<your-kaggle-api-key>"


# Connect and initialize the API
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# maxsize in bytes
datasets = kaggle.api.dataset_list(search="diabetes",max_size="10000000")
print(datasets)

# downloading from kaggle.com/kumargh/pimaindiansdiabetescsv'
# we write to the current directory path with './'
# api.dataset_download_files('kumargh/pimaindiansdiabetescsv',path='./')

#os.environ['KAGGLE_USERNAME'] = "konstantinosfilippou"
#os.environ['KAGGLE_KEY']      = "3514308d4ba9316c4f8b7bd9ecc245fb"

# List all metadata of the first in list info using the vars() function
ds      = datasets[1]
ds_vars = vars(ds)
for var in ds_vars:
    print(f"{var} = {ds_vars[var]}")


# Download and extract all files automated
#!mkdir dataset_download

# Download all files from the dataset in .zip / ds.ref is the path 'noahgift/social-power-nba'
api.dataset_download_files(ds.ref,path='./dataset_download')

# Unzipe the file to target path
# ./ means current directory

#ds.title.lower().replace(' ', '-')

# an example kaggle url is 'https://www.kaggle.com/datasets/mathchi/diabetes-data-set'
# take the last part of the url i.e the string diabetes-data-set

zip_name = ds.url.split('/')[-1]

with zipfile.ZipFile('./dataset_download/' + zip_name + '.zip','r') as zipref:
    zipref.extractall('./dataset_download')

#'/home/konstantinos/Documents/megaSyncfolder/master_files_mega_cloud/auto_ml/DEVOPS/pimaindiansdiabetescsv.zip'
# /home/konstantinos/Documents/megaSyncfolder/master_files_mega_cloud/auto_ml/DEVOPS/

# dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# List files that exist inside the zip 
files = kaggle.api.dataset_list_files(ds.ref).files
#files       = file_result.files
print(files)

# While loading we convert the files[0] file object to string
# dataset = loadtxt('./dataset_download/' + str(files[0]),delemiter=',')

dataframe = pd.read_csv(r'./dataset_download/' + str(files[0]))
dataset     = dataframe.values

# Delete the first row of the dataset - maybe are labels - if not we are losing one row 
dataset = np.delete(dataset,0,0)
input_shape = dataset.shape[1] - 1
X = dataset[:,0:input_shape]#.astype(float)
y = dataset[:,input_shape]
