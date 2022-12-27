# Documentation
# kaggle authentication
# Connect with kaggle database via its api and download automatically a dataset

# Libraries

import kaggle 
import zipfile
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from numpy                   import loadtxt


def p_kaggle(searchname):
    
    # Connect and initialize the API
    api = KaggleApi()
    api.authenticate()
         
    # Search a dataset via various criterions (maxsize in bytes)
    datasets = kaggle.api.dataset_list(search=searchname,max_size="10000000")
    #print(datasets)
    
    # List all metadata of the first in list info using the vars() function
    ds      = datasets[0]
    ds_vars = vars(ds)
    
    #for var in ds_vars:
    #    print(f"{var} = {ds_vars[var]}")
    
    global title,creator
    title   = ds.title
    creator = ds.creatorName
      
    #print(title,creator)
    
    # Download the zip file of the dataset in dataset_download folder (is created automatically)
    # ds.ref is the path of the kaggle dataset 'noahgift/social-power-nba'
    api.dataset_download_files(ds.ref,path='./dataset_download')
    
    # Unzip the zip     
    # -> the last part of the url has always the same format, so we use it to create the zip name.
    zip_name = ds.url.split('/')[-1] 

    with zipfile.ZipFile('./dataset_download/' + zip_name + '.zip','r') as zipref:
        zipref.extractall('./dataset_download')
        
    
    # List files that exist inside the zip 
    files = kaggle.api.dataset_list_files(ds.ref).files
    #print(files)
    
    #dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    filename = './dataset_download/' + str(files[0])
    #dataset = loadtxt(filename, delimiter=",")    
    
    # Process the csv file
    
    dataframe = pd.read_csv(r'./dataset_download/' + str(files[0]))
    dataset   = dataframe.values

    # Delete the first row of the dataset - maybe are labels - if not we are losing one row 
    dataset     = np.delete(dataset,0,0)
    input_shape = dataset.shape[1] - 1    
    
    X = dataset[:,0:input_shape].astype(float)
    y = dataset[:,input_shape]
    
    return X,y,input_shape
    
        