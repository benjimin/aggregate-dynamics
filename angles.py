import skyfield.api
import localindex
import dateutil.parser
import pyproj
import matplotlib.pyplot as plt

cell = 8, -41

ts = skyfield.api.load.timescale()
planets = skyfield.api.load('de421.bsp')

def get(*cell):
    records = localindex.get_time(*cell)

    times = [ts.utc(dateutil.parser.parse(i[1][0])) for i in records]

    projection = pyproj.Proj(init='epsg:3577')
    lon, lat = projection(*[(i + 0.5) * 100000 for i in cell], inverse=True)
    place = planets['earth'] + \
            skyfield.api.Topos(latitude_degrees=lat, longitude_degrees=lon)

    # why does skyfield not care about position datum?

    coords = [place.at(t).observe(planets['sun']).apparent().altaz() for t in times]

    altitude = [c[0]._degrees for c in coords]
    azimuth = [c[1]._degrees for c in coords]

    return altitude, azimuth

inc, angle = get(-13, -39) # Albany
plt.scatter(angle, inc, alpha=0.05, c='k')

inc, angle = get(8, -41) # Arapiles
plt.scatter(angle, inc, alpha=0.05, c='b')

inc, angle = get(11, -12) # Cape York
plt.scatter(angle, inc, alpha=0.05, c='r')

inc, angle = get(12, -49) # Port Arthur
plt.scatter(angle, inc, alpha=0.05, c='g')

plt.ylim([0, 90])
plt.xlim([20, 130])

incmin = min(i for i in inc)
