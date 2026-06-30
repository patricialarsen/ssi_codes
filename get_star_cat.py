
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import galsim
from math import floor

sys.path.insert(0,'/global/u2/p/plarsen/plarsen_git/dust_extinction')
import dust_extinction
sys.path.insert(0,'/global/u2/p/plarsen/plarsen_git/skyCatalogs/')
import skycatalogs


from skycatalogs.skyCatalogs import open_catalog

throughputs_dir = "/global/cfs/cdirs/descssim/imSim/lsst/data/throughputs_2023_09_07"
skycatalog_root = '/global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2/'
config_path = '/global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2/skyCatalog.yaml'


SEED = 2205802543 + 1
rng = np.random.default_rng(seed=SEED)

gal_data = pickle.load(open('data_file_WL.pkl','rb'))
ra_grid = np.load('ra_coords.npy')
dec_grid = np.load('dec_coords.npy')

permutation = np.arange(len(ra_grid))
rng.shuffle(permutation)
ra_grid = ra_grid[permutation]
dec_grid = dec_grid[permutation]

if len(gal_data['ra']) == len(ra_grid):
    gal_data['ra'] = ra_grid
else:
    print('WARNING: ra_grid stored in ra_coords.npy does not have the same size as data_file_WL.pkl')
if len(gal_data['dec']) == len(dec_grid):
    gal_data['dec'] = dec_grid
else:
    print('WARNING: dec_grid stored in dec_coords.npy does not have the same size as data_file_WL.pkl')

# Zeropoint recipe from /global/cfs/cdirs/descssim/imSim/skyCatalogs_v1.1.2/00README

bandpasses = {}
bands = "ugrizy"
for band in bands:
	bp_file = os.path.join(
		throughputs_dir,
		'baseline',
		f'total_{band}.dat',
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


n_gals_total = 0
n_gals_total_cuts = 0
n_stars_total = 0
n_stars_total_cuts = 0
star_ids_all = []
star_mag_i_all = []

# For every healpixel,
print('Scanning over healpixels. This can take a minute or so.')
print()
for hpix_id in hpix_ids_all:
    # Grab all the galaxies in this healpixel
    gal_catalog = cat.get_object_type_by_hp(hpix_id, 'diffsky_galaxy')
    gal_collections = gal_catalog.get_collections()

    for c in gal_collections:
        lsst_flux_i = c.get_native_attribute('lsst_flux_i')
        mag_i_lsst = bandpasses['i'].zeropoint - 2.5 * np.log10(lsst_flux_i)
        diskHalfLightRadiusArcsec = c.get_native_attribute('diskHalfLightRadiusArcsec')
        # A cut on maximum brightness helps reduce blending of SSI sources with each other.
        cuts = mag_i_lsst > 14.0
        cuts &= mag_i_lsst < 27.5
        cuts &= diskHalfLightRadiusArcsec < 10
        n_gals_total += len(cuts)
        n_gals_total_cuts += sum(cuts)

    # Grab all the stars in this healpixel
    star_catalog = cat.get_object_type_by_hp(hpix_id, 'star')
    star_collections = star_catalog.get_collections()

    # Count up the number of stars in this healpixel and add it to the total
    for c in star_collections:
        star_id = c.get_native_attribute('id')
        lsst_flux_i = c.get_native_attribute('lsst_flux_i')
        mag_i_lsst = bandpasses['i'].zeropoint - 2.5 * np.log10(lsst_flux_i)
        cuts = mag_i_lsst < 27.5
        # Up to here the cuts broadly follow the DES Y6 Balrog strategy,
        # revised for LSST mag limits.
        # For stars there is no associated half light radius in the catalog,
        # so no cut on that here.
        n_stars_total += len(cuts)
        n_stars_total_cuts += sum(cuts)
        star_ids_all.extend(list(star_id[cuts]))
        star_mag_i_all.extend(list(mag_i_lsst[cuts]))

star_to_galaxy_ratio = n_stars_total_cuts / n_gals_total_cuts

print()
print('Done scanning over healpixels.')
print(n_gals_total, 'total galaxies found.')
print(n_gals_total_cuts, 'total galaxies passing cuts.')
print(n_stars_total, 'total stars found.')
print(n_stars_total_cuts, 'total stars passing cuts.')
print('# stars passing cuts / # galaxies passing cuts =', star_to_galaxy_ratio)

n_sources_total = len(gal_data['galaxy_id'])
n_stars_to_inject = floor(star_to_galaxy_ratio * n_sources_total)
n_galaxies_to_inject = n_sources_total - n_stars_to_inject

def weight(m, m_ref):
    denom = 1. + np.exp(-4 * (m - m_ref))
    return 1. - 1. / denom

# should be the same as that used for the galaxies
w = weight(np.array(star_mag_i_all), m_ref=26.5)
p = w / np.sum(w) # Convert weights into normalized probabilities

star_data = {}
star_data['id'] = rng.choice(star_ids_all, size=n_stars_to_inject, replace=True, p=p)

# Initialize star_data based on the last -n_stars_to_inject rows of our ra--dec grid
for col in (['ra', 'dec'] + [f'mag_{band}_lsst' for band in 'ugrizy']):
    star_data[col] = gal_data[col][-n_stars_to_inject:]
# Restrict gal_data to only the first n_galaxies_to_inject rows
for col in gal_data:
    gal_data[col] = gal_data[col][:n_galaxies_to_inject]

pickle.dump(gal_data, open('data_file_gals_WL.pkl', 'wb'))
pickle.dump(star_data, open('data_file_stars_WL.pkl', 'wb'))
