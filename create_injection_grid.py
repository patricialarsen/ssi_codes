import numpy as np
import lsst.daf.butler as dafButler
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


# suggested values 


#spacing_arcsec = 20
#spacing_degrees = spacing_arcsec / 3600

# Should be slightly larger than the area of interest
#min_ra_degrees, max_ra_degrees = 37, 39
#min_dec_degrees, max_dec_degrees = 6, 8


def create_injection_grid(min_ra_degrees, max_ra_degrees, min_dec_degrees, max_dec_degrees, spacing_degrees, cut_bcg=False, cut_bcg_4bins=False):
   # min/max values should be slightly larger than the area of interest

    ra_span_degrees = max_ra_degrees - min_ra_degrees
    dec_span_degrees = max_dec_degrees - min_dec_degrees
    
    spacing_ra_dec0_degrees = float(spacing_degrees)
    spacing_dec_degrees = float(spacing_degrees * np.sqrt(3) / 2)

    spacing_ra_top_degrees = spacing_ra_dec0_degrees / np.cos(np.deg2rad(np.abs(max_dec_degrees)))
    N_across = int(ra_span_degrees // spacing_ra_top_degrees) + 1
    N_down = int(dec_span_degrees // spacing_dec_degrees) + 1

    ra_coords, dec_coords = [], []

    for idx in range(N_across):
        ra_relative_loc = spacing_ra_dec0_degrees * idx
    
        for idy in range(N_down):
            dec_relative_loc = spacing_dec_degrees * idy

            ra_relative_loc_shifted = ra_relative_loc
            if idy % 2 != 0:
                ra_relative_loc_shifted += spacing_ra_dec0_degrees / 2

            dec_coord = min_dec_degrees + dec_relative_loc
            ra_coord = min_ra_degrees + ra_relative_loc_shifted / np.cos(np.deg2rad(np.abs(dec_coord)))
            ra_coords.append(ra_coord)
            dec_coords.append(dec_coord)

    print('Sanity check: ', end='')
    if N_across * N_down == len(ra_coords):
        print(f'Number of computed grid points ({len(ra_coords)}) equals what we expect.')
    else:
        print('Warning: Number of computed grid points =', len(ra_coords),
              'which is not the value', N_across * N_down, 'derived above.')


    ra_coords = np.array(ra_coords)
    dec_coords = np.array(dec_coords)

    if cut_bcg:
        bcg_ra = 37.86501659859067
        bcg_dec = 6.982204815599694
        coords = SkyCoord(ra_coords*u.deg, dec_coords*u.deg)
        c2 = SkyCoord(bcg_ra*u.deg, bcg_dec*u.deg)
        sep_new = coords.separation(c2).to(u.deg).value

        ra_coords = ra_coords[sep_new<0.55]
        dec_coords = dec_coords[sep_new<0.55]

    if cut_bcg_4bins:
        bcg_ra = 37.86501659859067
        bcg_dec = 6.982204815599694
        coords = SkyCoord(ra_coords*u.deg, dec_coords*u.deg)
        c2 = SkyCoord(bcg_ra*u.deg, bcg_dec*u.deg)
        sep_new = coords.separation(c2).to(u.deg).value
        ra_coords = ra_coords[sep_new<0.2]
        dec_coords = dec_coords[sep_new<0.2]
    
    return ra_coords, dec_coords

if __name__ == "__main__":
    ra_coords, dec_coords = create_injection_grid(min_ra_degrees, max_ra_degrees, min_dec_degrees, max_dec_degrees, spacing_degrees)
    np.save('ra_coords.npy', ra_coords)
    np.save('dec_coords.npy', dec_coords)
