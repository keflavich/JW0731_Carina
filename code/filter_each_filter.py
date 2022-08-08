from filtering import estimate_background
import time
from astropy.io import fits
import os


basepath = '/orange/adamginsburg/jwst/jw02731/'


filenames = [
    'L3/t/jw02731-o001_t017_nircam_clear-f090w_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f187n_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f200w_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f335m_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_clear-f444w_i2d.fits',
    'L3/t/jw02731-o001_t017_nircam_f444w-f470n_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f1130w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f1280w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f1800w_i2d.fits',
    'L3/t/jw02731-o002_t017_miri_f770w_i2d.fits',
]

for fn_ in filenames:
    fn = f'{basepath}/{fn_}'

    fh = fits.open(fn)
    data = fh[1].data

    # metadata are only in header[0], but WCS are in header[1]
    header = fh[0].header
    header.update(fh[1].header)

    t0 = time.time()
    print(f"Started {fn} at t={t0:0.1f}")

    estimate_background(data, header, path_prefix=f'{basepath}/background_estimation/')

    print(f"Finished {fn} at t={time.time()-t0:0.1f}")
