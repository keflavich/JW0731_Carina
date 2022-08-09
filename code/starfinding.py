from scipy.ndimage import label, find_objects, center_of_mass, sum_labels 
import numpy as np
from scipy import ndimage
from astropy.table import Table

def is_star(data, labelmask):
    """
    Attempt to determine if a collection of blank pixels is actually a star by
    assuming the pixels closest to the center will be brighter than their
    surroundings
    """
    rind1 = ndimage.binary_dilation(labelmask).astype('bool')
    rind2 = ndimage.binary_dilation(rind1).astype('bool')
    rind2sum = data[rind2 &~ rind1].sum()
    rind1sum = data[rind1 &~ labelmask.astype('bool')].sum()
    return rind1sum > rind2sum

def finder_maker(max_size=100, min_size=0, min_sep_from_edge=20, *args, **kwargs):
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

        sources, nsources = label(saturated)
        slices = find_objects(sources)
        coms = center_of_mass(saturated, sources, np.arange(nsources))
        sizes = sum_labels(saturated, sources, np.arange(nsources))

        coms = np.array(coms)
        msfe = min_sep_from_edge

        sizes_ok = (sizes < max_size) & (sizes > min_size)
        coms_finite = np.isfinite(coms).all(axis=1)
        coms_inbounds = (
            (coms[:,1] > msfe) & (coms[:,0] > msfe) &
            (coms[:,1] < data.shape[1]-msfe) &
            (coms[:,0] < data.shape[0]-msfe)
        )
        all_ok = sizes_ok & coms_finite & coms_inbounds
        #is_star_ok = np.array([szok and is_star(data[slcs], sources[slcs] == srcid)
        #                       for srcid, (szok, slcs) in enumerate(tqdm(zip(all_ok, slices)))])
        #all_ok &= is_star_ok
        #is_star={is_star_ok.sum()},
        print(f"sizes={sizes_ok.sum()}, coms_finite={coms_finite.sum()}, coms_inbounds={coms_inbounds.sum()}, total={all_ok.sum()} candidates")


        tbl = Table()
        tbl['id'] = np.arange(1,all_ok.sum()+1)
        tbl['xcentroid'] = [cc[1] for cc, ok in zip(coms, all_ok) if ok]
        tbl['ycentroid'] = [cc[0] for cc, ok in zip(coms, all_ok) if ok]

        return tbl
    return saturated_finder        