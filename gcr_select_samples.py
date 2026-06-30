import numpy as np
import matplotlib.pyplot as plt
import sys
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pickle
import GCRCatalogs
GCRCatalogs.ConfigSource.set_config_source(dr=False)

# 
# some test values 
#
# SAMPLE_SIZE = len(np.load('ra_coords.npy'))
# SEED = 2205802543

def weight(m, m_ref):
    denom = 1. + np.exp(-4 * (m - m_ref))
    return 1. - 1. / denom


def select_from_gcr(SAMPLE_SIZE, SEED, weight=True, weight_mag=25.5, outname = 'data_file_WL.pkl', mag_choice = 'mag_i_lsst'):
    print('Loading GCRCatalog')
    catalog = GCRCatalogs.load_catalog('roman_rubin_2023_v1.1.2_elais')

    quantities = ['spheroidHalfLightRadius','diskHalfLightRadius','spheroidHalfLightRadiusArcsec','diskHalfLightRadiusArcsec',
                  'diskEllipticity1','diskEllipticity2','spheroidEllipticity1','spheroidEllipticity2',
                  'diskAxisRatio','spheroidAxisRatio','fknot','ra','dec','galaxy_id','redshift','redshiftHubble','mag_g_lsst',
                  'mag_r_lsst','mag_i_lsst','mag_u_lsst','mag_y_lsst','mag_z_lsst','um_source_galaxy_obs_sm','bulge_to_total_ratio']
    # cutting out too-large galaxies for overlap issues. 
    # cutting out too-bright and too-faint objects in i-band
    filters = ["mag_i_lsst > 14.0", "mag_i_lsst < 27.5", "diskHalfLightRadiusArcsec < 10"]

    # can take ~10 minutes to read so be prepared to wait
    print('Loading values from catalog. This can take awhile.')
    data = catalog.get_quantities(quantities, filters=filters)  

    rng = np.random.default_rng(seed=SEED)
    
    if weight==False:
        print('Selecting unweighted sample.')
        idx_vals = rng.choice(np.arange(len(data[mag_choice])), size=SAMPLE_SIZE, replace=True)

        dict_tot={}
        for key in data.keys():
            dict_tot[key] = data[key][idx_vals]

        pickle.dump(dict_tot, open(outname, 'wb'))
        print(outname, ' saved.')
    else:
        print('Selecting weighted sample.')
        w = weight(data[mag_choice], m_ref=25.5)
        p = w / np.sum(w) # Convert weights into normalized probabilities
        idx_vals = rng.choice(np.arange(len(data[mag_choice])), size=SAMPLE_SIZE, replace=True, p=p)

        dict_tot={}
        for key in data.keys():
            dict_tot[key] = data[key][idx_vals]

        pickle.dump(dict_tot, open(outname, 'wb'))
        print(outname, ' saved.')

if __name__ == "__main__":
    select_from_gcr(SAMPLE_SIZE, SEED, weight=True, weight_mag=26.5, outname = 'data_file_full.pkl', mag_choice = 'mag_i_lsst')

