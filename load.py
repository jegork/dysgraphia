import os
from zipfile import ZipFile
from glob import glob
import pandas as pd
import shutil

if '__name__' == '__main__':
    if not os.path.exists('dataSciRep_public/'):
        with ZipFile('Dysgraphia-detection-through-machine-learning/dataSciRep_public.zip', 'r') as f:
            f.extractall('./')
            print('Extracting archive')
    
    if not os.path.exists('data/'):
        print('Restructuring data')
        os.mkdir('data')

        data = {}

        for f in glob('dataSciRep_public/user0*'):
            curr_folder = os.path.join(f, 'session00001')
            filename = os.path.join(curr_folder, os.listdir(curr_folder)[0])
            filedata = pd.read_csv(filename, sep=' ', names=['x', 'y', 'time', 'on_surface', 'azimuth', 'altitude', 'pressure']).iloc[1:]
            filedata = filedata.astype('int')
            data[f.split('/')[1]] = filedata

        for k, v in data.items():
            v.to_csv(f'data/{k}.csv', index=False)

        shutil.rmtree('dataSciRep_public')