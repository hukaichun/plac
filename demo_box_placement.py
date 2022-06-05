from box_placement import Placement

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

half_width_and_length = [
    [2,1],
    [1,2],
    [1,1]
]

xy = [
    [0,0],
    [2,3],
    [3,3]
]

hls = np.asarray(half_width_and_length, dtype=float)
xy = np.asarray(xy, dtype=float)

placement = Placement(hls)
boxes = placement.get_box(xy, format="matplotlib")
rects = PatchCollection([Rectangle((x,y), w, h) for x,y,w,h in boxes], color='r')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.add_collection(rects)
ax.set_ylim([-10, 10])
ax.set_xlim([-10, 10])
ax.axis("equal")

plt.show()