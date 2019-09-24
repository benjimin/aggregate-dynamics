import skyfield.api
import localindex
import dateutil.parser
import pyproj
import matplotlib.pyplot as plt

cell = 8, -41

ts = skyfield.api.load.timescale()
planets = skyfield.api.load('de421.bsp')

records = localindex.get_time(*cell)

times = [ts.utc(dateutil.parser.parse(i[1][0])) for i in records]

#print(times)

projection = pyproj.Proj(init='epsg:3577')
lon, lat = projection(*[(i + 0.5) * 100000 for i in cell], inverse=True)
place = planets['earth'] + \
        skyfield.api.Topos(latitude_degrees=lat, longitude_degrees=lon)

# why does skyfield not care about position datum?

coords = [place.at(t).observe(planets['sun']).apparent().altaz() for t in times]

ra = [c[0]._degrees for c in coords]
dec = [c[1]._degrees for c in coords]



