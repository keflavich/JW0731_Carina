from scipy.ndimage import label, find_objects, center_of_mass, sum_labels
from tqdm.notebook import tqdm
import numpy as np
from scipy import ndimage
from astropy.table import Table
from astropy import table

def is_star(data, sources, srcid, slc, rindsize=3, min_flux=500, require_gradient=False):
    """
    Attempt to determine if a collection of blank pixels is actually a star by
    assuming the pixels closest to the center will be brighter than their
    surroundings
    """
    slc = tuple(slice(max(ss.start-rindsize, 0),
                      min(ss.stop+rindsize, shp)) for ss,shp in zip(slc, data.shape))

    labelmask = sources[slc] == srcid
    assert np.any(labelmask)

    rind1 = ndimage.binary_dilation(labelmask, iterations=2).astype('bool')
    rind2 = ndimage.binary_dilation(rind1, iterations=2).astype('bool')
    rind2sum = np.nansum(data[slc][rind2 & ~rind1])
    rind1sum = np.nansum(data[slc][rind1 & ~labelmask])

    rind3 = ndimage.binary_dilation(labelmask, iterations=rindsize)
    rind3sum = np.nansum(data[slc][rind3 & ~labelmask])

    return ((rind1sum > rind2sum) or not require_gradient) and rind3sum > min_flux

def finder_maker(max_size=100, min_size=0, min_sep_from_edge=20, min_flux=500,
                 rindsize=3, require_gradient=False, *args, **kwargs):
    """
    Create a saturated star finder that can select on the number of saturated pixels and the
    distance from the edge of the image
    """
    # criteria are based on examining some plots; they probably don't hold universally
    def saturated_finder(data,  *args, **kwargs):
        """
        Wrap the star finder to reject bad stars
        """
        saturated = (data==0)
        sources, nsources = label(saturated)
        if nsources == 0:
            raise ValueError("No saturated sources found")
        slices = find_objects(sources)

        coms = center_of_mass(saturated, sources, np.arange(nsources)+1)
        coms = np.array(coms)

        sizes = sum_labels(saturated, sources, np.arange(nsources)+1)
        msfe = min_sep_from_edge

        sizes_ok = (sizes < max_size) & (sizes > min_size)
        coms_finite = np.isfinite(coms).all(axis=1)
        coms_inbounds = (
            (coms[:,1] > msfe) & (coms[:,0] > msfe) &
            (coms[:,1] < data.shape[1]-msfe) &
            (coms[:,0] < data.shape[0]-msfe)
        )
        all_ok = sizes_ok & coms_finite & coms_inbounds
        is_star_ok = np.array([szok and is_star(data, sources, srcid+1, slcs, min_flux=min_flux, rindsize=rindsize)
                               for srcid, (szok, slcs) in enumerate(tqdm(zip(all_ok, slices)))])
        all_ok &= is_star_ok
        print(f"is_star={is_star_ok.sum()}, ", end="")
        print(f"sizes={sizes_ok.sum()}, coms_finite={coms_finite.sum()}, coms_inbounds={coms_inbounds.sum()}, total={all_ok.sum()} candidates")


        tbl = Table()
        tbl['id'] = np.arange(1,all_ok.sum()+1)
        tbl['xcentroid'] = [cc[1] for cc, ok in zip(coms, all_ok) if ok]
        tbl['ycentroid'] = [cc[0] for cc, ok in zip(coms, all_ok) if ok]

        return tbl
    return saturated_finder

def iteratively_remove_saturated_stars(data, header,
        fit_sizes=[251,101,101,51],
        nsaturated=[(100,500), (50,100), (30,50), (0,30)],
        min_flux=[1000, 1000, 1000, 1000],
        ap_rad=[15, 15, 15, 5],
        require_gradient=[False, False, False, True],
        dilations=[1,1,1,0],
        ):

    if header['INSTRUME'].lower() == 'nircam':
        psfgen = webbpsf.NIRCam()
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = webbpsf.MIRI()

    psfgen.filter = get_filtername(header)
    obsdate = header['DATE-OBS']
    psfgen.load_wss_opd_by_date(f'{obsdate}T00:00:00')
    big_grid = psfgen.psf_grid(num_psfs=16, all_detectors=False, fov_pixels=512)

    resid = data

    results = []

    for (minsz, maxsz), minflx, grad, fitsz, apsz, diliter in zip(nsaturated, min_flux, require_gradients, fit_size, ap_rad, dilations):
        phot = BasicPSFPhotometry(finder=finder_maker(min_size=minsz, max_size=maxsz, require_gradients=grad, min_flux=minflx),
                              group_maker=daogroup,
                              bkg_estimator=None, # must be none or it un-saturates pixels
                              #psf_model=epsf_model,
                              psf_model=big_grid,
                              fitter=LevMarLSQFitter(),
                              fitshape=fitsz,
                              aperture_radius=apsz*fwhm_pix)
        if diliter > 0:
            mask = ndimage.binary_dilation(data==0, iterations=diliter)
        else:
            mask = data==0

        result = phot(resid, mask=mask)
        results.append(result)

        resid = phot.get_residual_image()

    final_table = table.vstack(results)

    return final_table, resid
