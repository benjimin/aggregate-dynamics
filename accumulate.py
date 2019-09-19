import localindex
from datetime import date
import logging
import rasterio
import numpy as np
import tqdm

def load(filenames):
    for i, name in enumerate(filenames):
        assert name.endswith('.nc')
        with rasterio.open('NetCDF:' + name + ':water') as f:
            this = f.read(1)
        if i == 0:
            dest = this
        else: # fuser
            hole = (dest & 1).astype(np.bool)
            overlap = ~(hole | (this & 1).astype(np.bool))
            dest[hole] = this[hole]
            dest[overlap] |= this[overlap]
    return dest.astype(np.uint8)

def write(filename, month, x, y, data, nodata=None):
    """ Output raster to filesystem """
    filename = f'{filename}_{month}_{x}_{y}.tif'
    logging.info('Writing %s', filename)
    with rasterio.open(filename,
                       mode='w',
                       count=1,
                       dtype=data.dtype.name,
                       driver='GTIFF',
                       nodata=nodata,
                       tiled=True,
                       compress='LZW',  # balance IO volume and CPU speed
                       **profile) as destination:
            destination.write(data, 1)

def main(month, years, x, y):
    if month == 0:
        logging.info('all months')
    else:
        logging.info('month %i', month)
    logging.info('tile %i_%i', x, y)
    tile = x, y # e.g. arapiles: 8, -41

    logging.info('spanning %i years', years)
    t1 = date(2018, 1, 1)
    t0 = date(2018 - years, 1, 1)

    # filter observations by date
    obs = localindex.get(*tile)
    if month:
        obs = [fs for t, fs in obs if t.month == month and t0 < t < t1]
    else:
        obs = [fs for t, fs in obs if t0 < t < t1]
    logging.info('%i observations', len(obs))

    with rasterio.open('NetCDF:' + obs[0][0] + ':water') as template:
        global profile
        profile = dict(transform=template.profile['transform'],
                       crs=template.profile['crs'],
                       width=template.profile['width'],
                       height=template.profile['height'])

    data = (load(fs) for fs in obs) # lazy, not holding in memory

    land_or_sea = ~np.uint8(4) # bitmask

    for i, layer in enumerate(tqdm.tqdm(data, total=len(obs))):

        logging.debug('Layer %i', i)

        if not i: # initialise
            wet = np.zeros_like(layer, dtype=np.uint16)
            dry = np.zeros_like(wet)

            wet_sungraze = np.zeros_like(wet)
            dry_sungraze = np.zeros_like(wet)

        layer &= land_or_sea # unmask ocean

        wet += layer == 128
        dry += layer == 0

        wet_sungraze += layer == 8 + 128
        dry_sungraze += layer == 8

        del layer

    logging.info('Finalising statistics')

    clear = wet + dry
    #write('clear', month, x, y, clear)
    del dry

    clear_sungraze = wet_sungraze + dry_sungraze
    write('clearish', month, x, y, clear_sungraze)
    del dry_sungraze

    total = clear + clear_sungraze

    ratio = np.divide(clear_sungraze, total, dtype=np.float32)
    write('ratio', month, x, y, ratio, nodata=np.NaN)
    del ratio

    freq = np.divide(wet, clear, dtype=np.float32)
    #write('summary', month, x, y, freq, nodata=0)

    freq_sungraze = np.divide(wet_sungraze + wet, total, dtype=np.float32)
    #write('alternate', month, x, y, freq_sungraze, nodata=0)
    del wet_sungraze, wet, total

    #diff = freq_sungraze - freq
    diff = np.where(freq, -freq_sungraze, freq_sungraze) # ignore where known-wettable
    write('diff', month, x, y, diff, nodata=0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('month', type=int, choices=range(0, 13))
    parser.add_argument('years', type=int, choices=[1,2,3,4,5,10,15,100])
    parser.add_argument('x', type=int)
    parser.add_argument('y', type=int)
    config = parser.parse_args()
    main(config.month, config.years, config.x, config.y)