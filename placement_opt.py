from typing import Union

from box_placement import Placement

import numpy as np
from scipy.sparse import csr_matrix


class _PlacementInfo:
    def __init__(self, xy: np.ndarray, ref: Placement):
        assert xy.shape == ref.hls.shape
        self.__cross_section = ref.cross_section
        self.xy = xy

        self.__diff: np.ndarray = None
        self.__criteria: np.ndarray = None
        self.__x: np.ndarray = None
        self.__y: np.ndarray = None
        self.__xy: np.ndarray = None

    @property
    def diff(self):
        if self.__diff is None:
            xy = self.xy
            self.__diff = xy[:, np.newaxis] - xy[np.newaxis, ...]
        return self.__diff

    @property
    def criteria(self):
        if self.__criteria is None:
            abs_diff = np.abs(self.diff)
            cross_section = self.__cross_section
            criteria = cross_section[..., [1, 0]]*abs_diff
            self.__criteria = criteria > criteria[..., [1, 0]]
        return self.__criteria

    @property
    def _x(self):
        if self.__x is None:
            criteria = self.criteria
            self.__x = criteria[..., 0]
            self.__y = criteria[..., 1]
            self.__xy = ~(self.__x | self.__y)
            np.fill_diagonal(self.__xy, False)
        return self.__x

    @property
    def _y(self):
        if self.__y is None:
            self._x
        return self.__y

    @property
    def _xy(self):
        if self.__xy is None:
            self._x
        return self.__xy


class PlacementPyramid(Placement):
    def __init__(self, hls: np.ndarray, connect: Union[np.ndarray, csr_matrix]) -> None:
        super().__init__(hls)
        self.__connect = connect

    def __compute_info(self, xy: np.ndarray):
        return _PlacementInfo(xy, self)

    def __constraint(self, info: _PlacementInfo, compute_gradient=True):
        _x = info._x
        _y = info._y
        _xy = info._xy
        C_ij = np.zeros_like(_x, dtype=float)
        C_ij[_x] = 1 - np.abs(info.diff[_x, 0])/self.cross_section[_x, 0]
        C_ij[_y] = 1 - np.abs(info.diff[_y, 1]) / self.cross_section[_y, 1]
        C_ij[_xy] = .5*np.sum(1 - np.abs(info.diff[_xy]) /
                              self.cross_section[_xy], axis=-1)

        if not compute_gradient:
            return C_ij

        d_Cij = np.zeros_like(info.diff, dtype=float)
        d_Cij[_x, 0] = - np.sign(info.diff[_x, 0])/self.cross_section[_x, 0]
        d_Cij[_y, 1] = - np.sign(info.diff[_y, 1])/self.cross_section[_y, 1]
        d_Cij[_xy] = -.5*np.sign(info.diff[_xy]) / self.cross_section[_xy]

        return C_ij, d_Cij

    def __HPWL(self, info: _PlacementInfo, compute_gradient=True):
        node1, node2 = self.__connect.nonzero()
        weighted_diff = (info.xy[node1] - info.xy[node2])*self.__connect[node1, node2, np.newaxis]
        HPWL: float = np.sum(np.abs(weighted_diff))
        if not compute_gradient:
            return HPWL

        d_HPWL = np.zeros_like(info.diff)
        d_HPWL[node1, node2] = np.sign(weighted_diff)
        d_HPWL[node2, node1] = -d_HPWL[node1, node2]
        return HPWL, d_HPWL

    def __call__(self, xy: np.ndarray, compute_gradient=True):
        info = self.__compute_info(xy)
        return self.__HPWL(info, compute_gradient), self.__constraint(info, compute_gradient)


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    hls = np.asarray(
        [[1, 1],
         [2, 1],
         [1, 2]], dtype=float
    )

    n, dim = hls.shape

    xy = np.asarray(
        [[0,0],
         [1,0],
         [0,5]], dtype=float
    )
    connect = np.asarray(
        [[0, 1, 1],
         [0, 0, 1],
         [0, 0, 0]]
    )
    placement = PlacementPyramid(hls, connect)
    (HPWL, d_HPWL), (CONSTRAINT, d_CONSTRAINT) = placement(xy)
    d_HPWL = np.sum(d_HPWL, axis=1)
    print(CONSTRAINT, d_CONSTRAINT)

    d_CONSTRAINT = np.sum(d_CONSTRAINT, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([-10,10])
    ax.set_xlim([-10,10])

    boxes = placement.get_box(xy, "matplotlib")
    rects = [Rectangle((x,y), w, h) for x,y,w,h in boxes]
    ax.add_collection(PatchCollection(rects))

    total = d_HPWL+d_CONSTRAINT

    # ax.quiver(xy[:,0], xy[:, 1], -d_HPWL[:, 0], -d_HPWL[:, 1], color='b')
    ax.quiver(xy[:,0], xy[:, 1], -d_CONSTRAINT[:, 0], -d_CONSTRAINT[:, 1], color='r')
    ax.quiver(xy[:,0], xy[:, 1], -total[:,0], -total[:, 1], color="k")    

    plt.show()

