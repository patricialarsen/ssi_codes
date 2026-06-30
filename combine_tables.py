from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn, join, vstack
from astropy import units as u
import numpy as np
import glob



for band in 'g','r','i','z':
    file_list = glob.glob('iterative_run/table_out_band_'+band+'.csv')
    file_list += glob.glob('tables_full/table_out_stars_band_'+band+'.csv')
    file_list = np.sort(file_list)
    id_count = 0 
    group_count = 0 

    data_joined = ascii.read(file_list[0])
    id_count+= np.max(data_joined['injection_id'])+1
    group_count += np.max(data_joined['group_id'])+1


    for i in range(len(file_list)-1):
        data = ascii.read(file_list[i+1])
        id_count_iter = np.max(data['injection_id'])+1
        group_count_iter = np.max(data['group_id'])+1

        data['injection_id'] = data['injection_id'] + id_count
        data['group_id'] = data['group_id'] + group_count
        id_count += id_count_iter
        group_count += group_count_iter

        data_joined = vstack([data_joined, data])

    new_col = data_joined['source_type'].astype(f"<U{13}")
    new_col[new_col!='Sersic']= 'DeltaFunction'
    data_joined['source_type'] = new_col

    new_col = data_joined['n'].astype(float)
    data_joined['n'] = new_col
    new_col = data_joined['galaxy_id'].astype(float)
    data_joined['galaxy_id'] = new_col
    data_joined = data_joined.filled(np.nan)
    data_joined.write('table_out_band_'+band+'_total.csv',overwrite=True)
    print('data written')
