from filtering import estimate_background
from filtering import get_filtername, get_fwhm
from starfinding import iteratively_remove_saturated_stars
import time
from astropy.io import fits
from astropy import wcs
import os
import regions


basepath = '/orange/adamginsburg/jwst/jw02731/'


filenames = [
    'L3/t/jw02731-o001_t017_nircam_f444w-f470n_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f187n_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f090w_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f444w_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f335m_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f200w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f1130w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f1280w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f1800w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f770w_i2d.fits',
]


import regions
rois = regions.Regions.read('/orange/adamginsburg/jwst/jw02731/ROIs.reg')
r0, r1 = rois


for fn_ in filenames:
    fn = f'{basepath}/{fn_}'
    path_prefix = f'{basepath}/background_estimation_cutout/'

    fh = fits.open(fn)
    data = fh[1].data

    # metadata are only in header[0], but WCS are in header[1]
    header = fh[0].header
    header.update(fh[1].header)
    ww = wcs.WCS(fh[1].header)
    mask = (r0|r1).to_pixel(ww).to_mask()
    bigslc, smlslc = mask.bbox.get_overlap_slices(data.shape)
    ww = ww[bigslc]
    data = data[bigslc]
    header.update(ww.to_header())

    t0 = time.time()
    print(f"Started {fn} saturated at t={t0:0.1f}")

    saturated_table, saturated_removed = iteratively_remove_saturated_stars(data, header)

    filtername = get_filtername(header)

    fits.PrimaryHDU(data=saturated_removed, header=header).writeto(f'{path_prefix}/{filtername}_saturated_stars_removed.fits', overwrite=True)
    saturated_table.write(f'{path_prefix}/{filtername}_saturated_stars_catalog.fits', overwrite=True)

    print(f"Started {fn} background at t={t0:0.1f}")

    estimate_background(saturated_removed, header, path_prefix=path_prefix)

    print(f"Finished {fn} at t={time.time()-t0:0.1f}")
