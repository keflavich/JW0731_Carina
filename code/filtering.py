import photutils
import webbpsf
from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry
import numpy as np
import time
from astropy.stats import mad_std
from astropy import stats
from astropy.io import fits
from astropy import wcs
from astropy import log
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy import units as u
from astropy.table import Table
from astropy.nddata import NDData
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import table
from tqdm.notebook import tqdm
from astroquery.svo_fps import SvoFps
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

import os
os.environ['WEBBPSF_PATH'] = '/orange/adamginsburg/jwst/webbpsf-data/'
import webbpsf

if __name__ == "__main__":
    im1 = fits.open('/orange/adamginsburg/jwst/jw02731/L3/t/jw02731-o001_t017_nircam_clear-f444w_i2d.fits')
    data = im1[1].data
    basetable = Table.read('/orange/adamginsburg/jwst/jw02731/L3/t/jw02731-o001_t017_nircam_clear-f444w_cat.ecsv')



def get_fwhm(header, instrument_replacement='NIRCam'):
    """
    Paramters
    ---------
    header : fits.Header
        The header of the file of interest
    instrument_replacement : str
        Case-sensitive version of instrument name

    Returns
    -------
    fwhm : u.Quantity
        The FWHM in arcseconds
    fwhm_pix : float
        The FWHM in pixels
    """

    instrument = header['INSTRUME']
    telescope = header['TELESCOP']
    filt = header['FILTER']
    wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt}')
    filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
    filter_table.add_index('filterID')
    eff_wavelength = filter_table.loc[f'{telescope}/{instrument_replacement}.{filt}']['WavelengthEff'] * u.AA

    fwhm = (1.22 * eff_wavelength / (6.5*u.m)).to(u.arcsec, u.dimensionless_angles())

    ww = wcs.WCS(header)
    pixscale = ww.proj_plane_pixel_area()**0.5
    fwhm_pix = (fwhm / pixscale).decompose().value

    return fwhm, fwhm_pix


def estimate_background(data, header, medfilt_size=[15,15], do_segment_mask=False, save_products=True,
                        psf_size=31, nsigma_threshold=10):
    """
    """

    fwhm, fwhm_pix = get_fwhm(header)
    filtername = header['FILTER']

    t0 = time.time()

    medfilt = median_filter(data, size=medfilt_size)
    log.info(f"Median filter done in {time.time()-t0:0.1f}s")


    medfilt_sub = data - medfilt

    if do_segment_mask:
        # this was an exploration to see if segmentation-based-masking was a good idea.
        # It's not.
        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)


        threshold = detect_threshold(medfilt_sub, nsigma=5.0, sigma_clip=sigma_clip)
        segment_img = detect_sources(medfilt_sub, threshold, npixels=10)
        footprint = circular_footprint(radius=2)
        segment_mask = segment_img.make_source_mask(footprint=footprint)
        mean, median, std = sigma_clipped_stats(medfilt_sub, sigma=3.0, mask=segment_mask)


    # make webbpsf
    nc = webbpsf.NIRCam()
    nc.filter = filtername


    # mask out the pixels that are saturated
    medfilt_sub[data==0] = np.nan

    # calculate a PSF and use it as a kernel
    psf_kernel = nc.calc_psf(fov_pixels=psf_size, oversample=1)[0].data
    log.info(f"Calculating PSF kernel done at {time.time()-t0:0.1f}s")

    # replace the saturated pixels with interpolated ones
    datafilt_conv_psf = (convolve(medfiltsub, psf_kernel, nan_treatment='interpolate'))
    log.info(f"Convolution done at {time.time()-t0:0.1f}s")

    filled_in_pixels = (data==0) & ~np.isnan(datafilt_conv_psf)

    datafilt_conv_psf = np.nan_to_num(datafilt_conv_psf)


    # create a PSF-based mask to mask out diffraction spikes
    # (the psf_size+1 here is a bit of a hack that might break if you try to put in different PSF sizes...)
    pp = nc.calc_psf(oversample=1, fov_pixels=psf_size+1)[0].data
    log.info(f"Calculating PSF for mask done at {time.time()-t0:0.1f}s")

    # total guess heuristics...
    psfmask1000 = pp > 0.001*pp.max()
    psfmask100 = pp > 0.01*pp.max()
    psfmask5pct = pp > 0.05*pp.max()


    # find stars to mask out
    err_est_conv = stats.mad_std(datafilt_conv_psf)
    daofind_deep = DAOStarFinder(threshold=10 * err_est_conv, fwhm=fwhm_pix*2**0.5, roundhi=0.25, roundlo=-1.0,
                                sharplo=0.30, sharphi=1.40)

    stars_deep_conv = daofind_deep(datafilt_conv_psf)

    # allow any shape at all, but strong cut on S/N
    daofind_shallow_conv = DAOStarFinder(threshold=250 * err_est_conv, fwhm=fwhm_pix, roundhi=4, roundlo=-1.0,
                                sharplo=0.001, sharphi=4.40)

    stars_shallow_conv = daofind_shallow_conv(datafilt_conv_psf)
    log.info(f"Starfinding for mask done at {time.time()-t0:0.1f}s.  n_shallow={len(stars_shallow_conv)}, n_deep={len(stars_deep_conv)}")


    # Use the original data, unfiltered and unconvolved, because we're masking out the stars
    masked_data = data.copy()
    for row in stars_deep_conv:
        xc,yc = row['xcentroid'], row['ycentroid']
        #mreg = regions.CirclePixelRegion(regions.PixCoord(xc, yc), radius=fwhm_pix*1.5)
        mreg = regions.RectanglePixelRegion(regions.PixCoord(xc, yc), width=psf_size, height=psf_size)

        msk = mreg.to_mask()
        slcs, sslcs = msk.get_overlap_slices(masked_data.shape)
        try:
            #masked_data[slcs][msk.data.astype('bool')] = np.nan
            masked_data[slcs][psfmask5pct.astype('bool')[sslcs]] = np.nan

        except IndexError:
            # border case
            pass
    for row in stars_shallow_conv:
        xc,yc = row['xcentroid'], row['ycentroid']
        #mreg = regions.CirclePixelRegion(regions.PixCoord(xc, yc), radius=15)
        mreg = regions.RectanglePixelRegion(regions.PixCoord(xc, yc), width=psf_size, height=psf_size)
        msk = mreg.to_mask()
        slcs, sslcs = msk.get_overlap_slices(masked_data.shape)
        try:
            #masked_data[slcs][msk.data.astype('bool')] = np.nan
            masked_data[slcs][psfmask1000.astype('bool')[sslcs]] = np.nan
        except IndexError:
            # border case
            pass


    # commented out experiments
    #medfilt_masked = median_filter(masked_data, size=[9,9])
    #conv = convolve(medfilt_masked, kernel=Gaussian2DKernel(fwhm_pix), nan_treatment='interpolate')
    #medfilt_masked = median_filter(medfilt_masked, size=[9,9])

    # now that we've masked out the stars, we fill back in the star positions by interpolating into them by smoothing the background
    conv = convolve(masked_data, kernel=Gaussian2DKernel(fwhm_pix, x_size=np.ceil(10*fwhm_pix)), nan_treatment='interpolate')
    log.info(f"Masked convolution done at {time.time()-t0:0.1f}s")

    #medfilt_masked[np.isnan(medfilt_masked)] = conv[np.isnan(medfilt_masked)]

    # now we can replace the nans in the original data
    data_replacenans = data.copy()
    data_replacenans[filled_in_pixels] = datafilt_conv_psf[filled_in_pixels]

    # and then subtract off our star-masked convolved image
    # filtered_data = data_replacenans - medfilt_masked
    filtered_data = data_replacenans - conv


    if save_products:

        # fits.PrimaryHDU(data=medfilt_masked, header=im1[1].header).writeto('F444W_filter-based-background.fits', overwrite=True)

        fits.PrimaryHDU(data=conv, header=header).writeto(f'{filtername}_convolution-based-background.fits', overwrite=True)
        fits.PrimaryHDU(data=filtered_data, header=header).writeto(f'{filtername}_filter-based-background-subtraction.fits', overwrite=True)



    # create an empirical PSF
    data_cts = filtered_data / header['PHOTMJSR']

    bad_shallow = (np.abs(stars_shallow_conv['roundness2']) > 0.25) | (np.abs(stars_shallow_conv['roundness1']) > 0.25)

    stars_tbl = Table()
    stars_tbl['x'] = stars_shallow_conv['xcentroid'][~bad_shallow]
    stars_tbl['y'] = stars_shallow_conv['ycentroid'][~bad_shallow]

    nddata = NDData(data=filtered_data)
    sz = psf_size
    stars_ = extract_stars(nddata, stars_tbl, size=sz)

    # Remove off-center stars
    stars = photutils.psf.epsf_stars.EPSFStars([star for star in stars_
                                                if np.unravel_index(np.argmax(star), star.shape) == (sz//2, sz//2)
                                                and data[int(star.center[1]), int(star.center[0])] > 0 # don't want the ones we replaced
                                               ])

    log.info(f"EPSF calculation beginning at {time.time()-t0:0.1f}s")
    epsf_builder = EPSFBuilder(oversampling=4, maxiters=3, smoothing_kernel='quadratic')

    epsf_quadratic_filtered, fitted_stars = epsf_builder(stars)
    log.info(f"EPSF calculation done at {time.time()-t0:0.1f}s")


    # ## Do the PSF photometry
    # 
    # DAOGroup decides which subset of stars needs to be simultaneously fitted together - i.e., it deals with blended sources.
    daogroup = DAOGroup(5 * fwhm_pix)
    mmm_bkg = MMMBackground()

    filtered_errest = stats.mad_std(filtered_data, ignore_nan=True)

    daofind_fin = DAOStarFinder(threshold=nsigma_threshold * filtered_errest, fwhm=fwhm_pix, roundhi=1.0, roundlo=-1.0,
                                sharplo=0.30, sharphi=1.40)
    finstars = daofind_fin(filtered_data)
    log.info(f"First-pass starfinding calculation done at {time.time()-t0:0.1f}s.  Found {len(finstars}) stars.")

    # criteria are based on examining some plots; they probably don't hold universally
    def filtered_finder(data, *args, **kwargs):
        """
        Wrap the star finder to reject bad stars
        """
        finstars = daofind_fin(data)
        bad = ((finstars['roundness1'] > finstars['mag']*0.4/8+0.65) | (finstars['roundness1'] < finstars['mag']*-0.4/8-0.5) | 
               (finstars['sharpness'] < 0.48) | (finstars['sharpness'] > 0.6) | 
               (finstars['roundness2'] > finstars['mag']*0.4/8+0.55) | (finstars['roundness2'] < finstars['mag']*-0.4/8-0.5))
        finstars = finstars[~bad]
        finstars['id'] = np.arange(1, len(finstars)+1)

        # this will print at each iteration
        log.info(f"Filtered {bad.sum()} bad stars out.  t={time.time()-t0:0.1f}")

        return finstars

    phot = IterativelySubtractedPSFPhotometry(finder=filtered_finder, group_maker=daogroup,
                                              bkg_estimator=mmm_bkg,
                                              #psf_model=psf_modelgrid[0],
                                              psf_model=epsf_quadratic_filtered,
                                              fitter=LevMarLSQFitter(),
                                              niters=2, fitshape=(11, 11), aperture_radius=2*fwhm_pix)





    # operate on the full data
    log.info(f"Doing full photometry.  t={time.time()-t0:0.1f}")
    result_full = phot(np.nan_to_num(filtered_data))
    log.info(f"Done with full photometry.  t={time.time()-t0:0.1f}")
    resid = phot.get_residual_image()
    log.info(f"Done with final residual estimate.  t={time.time()-t0:0.1f}")
    if save_products:
        result_full.write(f'{filtername}_fullfield_ePSF_photometry.ecsv', overwrite=True)
        fits.PrimaryHDU(data=resid, header=header).writeto(f'{filtername}_psfphot_stars_removed.fits', overwrite=True)

    resid_orig = photutils.psf.utils.subtract_psf(data, epsf_quadratic_filtered, result_full)
    log.info(f"Done with final star subtraction from original data.  t={time.time()-t0:0.1f}")
    if save_products:
        fits.PrimaryHDU(data=resid_orig, header=header).writeto(f'{filtername}_psfphot_stars_removed.fits', overwrite=True)

    resid_orig_filled = photutils.psf.utils.subtract_psf(datafilt_conv_psf, epsf_quadratic_filtered, result_full)
    log.info(f"Done with final star subtraction from original filled in data.  t={time.time()-t0:0.1f}")
    if save_products:
        fits.PrimaryHDU(data=resid_orig_filled, header=header).writeto(f'{filtername}_psfphot_stars_filled_then_removed.fits', overwrite=True)
