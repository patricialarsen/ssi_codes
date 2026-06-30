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

_worker_state = {}

def init_worker(gal_data_idx_list_, gal_id_to_catalog_gal_idx_, gal_collection_):
    _worker_state["gal_data_idx_list"] = gal_data_idx_list_
    _worker_state["gal_id_to_catalog_gal_idx"] = gal_id_to_catalog_gal_idx_
    _worker_state["gal_collection"] = gal_collection_

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

SEED = 4205802543 + 2
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
gal_data = pickle.load(open('iterative_run/data_file_gals_WL.pkl','rb'))


def integrate_sed_gal(iterator_idx):
    gal_data_idx_list = _worker_state["gal_data_idx_list"]
    gal_id_to_catalog_gal_idx = _worker_state["gal_id_to_catalog_gal_idx"]
    gal_collection = _worker_state["gal_collection"]

    gal_data_idx = gal_data_idx_list[iterator_idx]
    gal_id = gal_data['galaxy_id'][gal_data_idx]
    c_idx = gal_id_to_catalog_gal_idx[gal_id]
    galaxy = gal_collection[c_idx]

    sed_bulge = galaxy._get_sed('bulge')
    sed_disk = galaxy._get_sed('disk')
    
    g1, g2, mu = galaxy.get_wl_params()
    sed_bulge = sed_bulge * mu
    sed_disk = sed_disk * mu

    ra, dec = gal_data['ra'][gal_data_idx], gal_data['dec'][gal_data_idx]
    mwAv_shifted = make_MW_extinction_av(ra, dec)
    MWE = MilkyWayExtinction()
    sed_bulge = MWE.extinguish(sed_bulge, mwAv_shifted)
    sed_disk = MWE.extinguish(sed_disk, mwAv_shifted)

    data_gal_integrated_sed = {'ra': ra, 'dec': dec}
    for col in ['spheroidEllipticity1', 'spheroidEllipticity2',
               'diskEllipticity1', 'diskEllipticity2',
               'spheroidHalfLightRadiusArcsec', 'diskHalfLightRadiusArcsec',
               'shear1', 'shear2', 'convergence', 'galaxy_id']:
        data_gal_integrated_sed[col] = galaxy.get_native_attribute(col)
    
    for band, bandpass in bandpasses.items():
        data_gal_integrated_sed['disk_mag_' + band] = sed_disk.calculateMagnitude(bandpass)
        data_gal_integrated_sed['bulge_mag_' + band] = sed_bulge.calculateMagnitude(bandpass)
        data_gal_integrated_sed['total_mag_' + band] = (sed_disk + sed_bulge).calculateMagnitude(bandpass)

    return data_gal_integrated_sed


integrated_gal_tables_hpix={}

for hpx_idx, hpix_id in enumerate(hpix_ids_all):
    
    gal_catalog = cat.get_object_type_by_hp(hpix_id, 'diffsky_galaxy')
    gal_collections = gal_catalog.get_collections()

    integrated_gal_tables={}
    for collection_idx in range(3):
        gal_collection = gal_collections[collection_idx]

        gal_id_to_catalog_gal_idx = {}

        print('Matching galaxies by galaxy_id')
        gal_ids = gal_collection.get_native_attribute('galaxy_id')

        # Be able to look up galaxy SEDs by gal_id
        c_idx_list = np.where(np.isin(gal_ids, gal_data['galaxy_id']))[0]
        print('Number of galaxies from this collection in our data set:', len(c_idx_list))
        for c_idx in c_idx_list:
            gal_id_to_catalog_gal_idx[gal_ids[c_idx]] = c_idx

        # Focus on just rows in our SSI table that are matched to gal_ids in this collection
        gal_data_idx_list = list(np.where(np.isin(gal_data['galaxy_id'], gal_ids))[0])
        print('Number of rows in our injection sample with matches in this collection:', len(gal_data_idx_list))


        N_START = 0
        N_ROWS = len(gal_data_idx_list)
        if N_ROWS == 0:
            print(f"No matching galaxies for hpix {hpix_id}, collection {collection_idx}; skipping")
            continue

        with Pool(
            N_PROCESSES,
            initializer=init_worker,
            initargs=(gal_data_idx_list, gal_id_to_catalog_gal_idx, gal_collection),
        ) as p:
            sed_integrated_gal_list = p.map(integrate_sed_gal,range(N_START, N_ROWS))
    
        sed_integrated_gal_all = astropy.table.Table(sed_integrated_gal_list)
        integrated_gal_tables[collection_idx] = sed_integrated_gal_all

        if len(sed_integrated_gal_all) == N_ROWS:
            print('SED-integrated galaxy table has all the rows we expect')
        else:
            print('ERROR: SED-integrated galaxy table is missing something')

    if len(integrated_gal_tables) > 0:
        sed_integrated_gal_all = vstack(list(integrated_gal_tables.values()), join_type='exact')
        integrated_gal_tables_hpix[hpx_idx] = sed_integrated_gal_all
    else:
        print(f"No matching galaxies for hpix {hpix_id}; skipping")


sed_integrated_gal_all = vstack([integrated_gal_tables_hpix[k] for k in integrated_gal_tables_hpix],join_type='exact')
n_gals = len(sed_integrated_gal_all['ra'])

# Repeating gives group 0,0,1,1,2,2 etc.
group_id = np.repeat(np.arange(n_gals), 2)
source_types = np.repeat('Sersic', n_gals * 2)
# Tiling gives interspersing 1,4,1,4 Sersic index,
# for the disk (n=1) and spheroid/bulge (n=4) components of each galaxy
n = np.tile([1, 4], n_gals) 

hlr_disk = sed_integrated_gal_all['diskHalfLightRadiusArcsec']
hlr_spheroid = sed_integrated_gal_all['spheroidHalfLightRadiusArcsec']
hlr = np.zeros(n_gals * 2)
hlr[0::2] = hlr_disk
hlr[1::2] = hlr_spheroid

e1_disk = sed_integrated_gal_all['diskEllipticity1']
e2_disk = sed_integrated_gal_all['diskEllipticity2']
e1_spheroid = sed_integrated_gal_all['spheroidEllipticity1']
e2_spheroid = sed_integrated_gal_all['spheroidEllipticity2']
e1s = np.zeros(n_gals * 2)
e1s[0::2] = e1_disk
e1s[1::2] = e1_spheroid
e2s = np.zeros(n_gals * 2)
e2s[0::2] = e2_disk
e2s[1::2] = e2_spheroid

s1 = np.repeat(sed_integrated_gal_all['shear1'], 2)
s2 = np.repeat(sed_integrated_gal_all['shear2'], 2)
ra = np.repeat(sed_integrated_gal_all['ra'], 2)  
dec = np.repeat(sed_integrated_gal_all['dec'], 2)

gal_id = np.repeat(sed_integrated_gal_all['galaxy_id'], 2)

beta = np.repeat(rng.uniform(0, 180, n_gals), 2)


data_out = {
    "injection_id": np.arange(n_gals * 2),
    "group_id": group_id,
    "ra": ra,
    "dec": dec,
    "source_type": source_types,
    "n": n,
    "half_light_radius": hlr,
    "e1": e1s,
    "e2": e2s,
    "g1":s1,
    "g2":s2,
    "beta": beta,
    "galaxy_id": gal_id,
}

def make_gal_table_for_band(band, sed_integrated_gal_all, data_out):
    mag_disk = sed_integrated_gal_all['disk_mag_' + band]
    mag_spheroid = sed_integrated_gal_all['bulge_mag_' + band]
    mags = np.zeros(len(mag_disk) * 2)
    mags[0::2] = mag_disk
    mags[1::2] = mag_spheroid
    
    data_out['mag'] = mags
    table_out_gal = astropy.table.Table(data_out)
    
    return table_out_gal

for band in bands:
    print(band, 'band')
    table_out_gal_band = make_gal_table_for_band(
        band, sed_integrated_gal_all, data_out
    )
    table_out_gal_band.write('iterative_run/table_out_band_'+band +'.csv', overwrite=True)
                             

