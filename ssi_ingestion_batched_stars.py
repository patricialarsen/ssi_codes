# Imports work with desc-stack-weekly kernel (as of 27 August 2025)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import astropy
import os
import sys
import galsim
from multiprocessing import Pool
from math import floor
from astropy.table import vstack, Table

sys.path.insert(0,'/global/u2/p/plarsen/plarsen_git/dust_extinction')
import dust_extinction
sys.path.insert(0,'/global/u2/p/plarsen/plarsen_git/skyCatalogs/')
import skycatalogs


from skycatalogs.skyCatalogs import open_catalog
from skycatalogs.utils.sed_tools import MilkyWayExtinction
from skycatalogs.utils.creator_utils import make_MW_extinction_av

os.environ["SIMS_SED_LIBRARY_DIR"]='/global/cfs/cdirs/descssim/imSim/lsst/data/sims_sed_library/'
throughputs_dir = "/global/cfs/cdirs/descssim/imSim/lsst/data/throughputs_2023_09_07"
skycatalog_root = '/global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2/'
config_path = '/global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2/skyCatalog.yaml'

SEED = 2205802543 + 2
rng = np.random.default_rng(seed=SEED)

# Zeropoint recipe from /global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2/00README

N_PROCESSES = 128

bandpasses = {}
bands = "ugrizy"
for band in bands:
	bp_file = os.path.join(
		throughputs_dir,
		'baseline',
		'total_'+str(band)+'.dat',
	)
	lut = galsim.LookupTable.from_file(bp_file)
	bp = galsim.Bandpass(lut, wave_type='nm')
	bp = bp.truncate(relative_throughput=1e-3)
	bp = bp.thin()
	bp = bp.withZeropoint('AB')
	bandpasses[band] = bp

# All healpixels available in /global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2
hpix_ids_all = [10050, 10051, 10052, 10053, 10177, 10178, 10179,
                10180, 10181, 10305, 10306, 10307, 10308, 10429,
                10430, 10431, 10432, 10549, 10550, 10551, 10552,
                10665, 10666, 10667, 10668, 10777, 10778, 10779,
                9921, 9922, 9923, 9924, 9925]


cat = open_catalog(config_path, skycatalog_root=skycatalog_root)

star_data = pickle.load(open('data_file_stars_WL.pkl','rb'))

def integrate_sed_star(iterator_idx):
    star_data_idx = star_data_idx_list[iterator_idx]
    star_id = star_data['id'][star_data_idx]
    c_idx = star_id_to_catalog_star_idx[star_id]
    star = star_collection[c_idx]

    ra, dec = star_data['ra'][star_data_idx], star_data['dec'][star_data_idx]
    mwAv_shifted = make_MW_extinction_av(ra, dec)
    MWE = MilkyWayExtinction()
    sed = star._get_sed()
    sed = MWE.extinguish(sed, mwAv_shifted)

    data_star_integrated_sed = {'ra': ra, 'dec': dec, 'id': star_id}
    for band, bandpass in bandpasses.items():
	    data_star_integrated_sed['mag_' + band] = sed.calculateMagnitude(bandpass)

    return data_star_integrated_sed


sed_integrated_star_all_pixels = []

for i in range(len(hpix_ids_all)):
    hpix_id = hpix_ids_all[i]

    star_catalog = cat.get_object_type_by_hp(hpix_id, 'star')
    star_collections = star_catalog.get_collections()

    # For stars, there is only ever one collection,
    # while for galaxies there are generally three.
    star_collection = star_collections[0]

    star_id_to_catalog_star_idx = {}

    print('Matching stars by id')
    star_ids = star_collection.get_native_attribute('id')

    # Be able to look up star SEDs by star_id
    c_idx_list = np.where(np.isin(star_ids, star_data['id']))[0]
    print('Number of stars from this collection in our data set:', len(c_idx_list))
    for c_idx in c_idx_list:
        star_id_to_catalog_star_idx[star_ids[c_idx]] = c_idx

    # Focus on just rows in our SSI table that are matched to star_ids in this collection
    star_data_idx_list = list(np.where(np.isin(star_data['id'], star_ids))[0])
    print('Number of rows in our injection sample with matches in this collection:', len(star_data_idx_list))

    N_START = 0
    N_ROWS = len(star_data_idx_list)

    if N_ROWS == 0:
        print(f'No matched stars for hpix {hpix_id}; skipping')
        continue

    with Pool(N_PROCESSES) as p:
        sed_integrated_star_list = p.map(integrate_sed_star, range(N_START,N_ROWS))

    sed_integrated_star_pixel = astropy.table.Table(sed_integrated_star_list)

    if len(sed_integrated_star_pixel) == N_ROWS:
        print('SED-integrated star table has all the rows we expect')
    else:
        print('ERROR: SED-integrated star table is missing something')

    sed_integrated_star_pixel['hpix_id'] = hpix_id
    sed_integrated_star_all_pixels.append(sed_integrated_star_pixel)

# Combine all healpixels into one table
sed_integrated_star_all = vstack(sed_integrated_star_all_pixels)

n_stars = len(sed_integrated_star_all)
source_types = np.repeat('PointSource', n_stars)

ra = sed_integrated_star_all['ra']
dec = sed_integrated_star_all['dec']

data_out = {
        'injection_id': np.arange(n_stars),
        'group_id': np.arange(n_stars),
        'ra': ra,
        'dec': dec,
        'source_type':source_types}

def make_star_table_for_band(band, sed_integrated_star_all, data_out):
    data_out['mag'] = sed_integrated_star_all['mag_' + band]
    table_out_gal = astropy.table.Table(data_out)
    
    return table_out_gal

for band in bands:
    print(band, 'band')
    table_out_star_band = make_star_table_for_band(
        band, sed_integrated_star_all, data_out
    )
    table_out_star_band.write('table_out_stars_band_'+band+'.csv',overwrite=True)
