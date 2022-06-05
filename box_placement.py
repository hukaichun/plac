import numpy as np


class Placement:
    def __init__(self, hls:np.ndarray) -> None:
        self.__hls = hls
        
        self.__cross_section:np.ndarray = None

    @property
    def cross_section(self)->np.ndarray:
        if self.__cross_section is None:
            self.__cross_section = self.__hls[:, np.newaxis] + self.__hls[np.newaxis, ...]
        return self.__cross_section

    @property
    def hls(self):
        return self.__hls

    def get_box(self, xy:np.ndarray, format:str="shapely"):
        assert xy.shape == self.__hls.shape

        if format == "shapely":
            lp = xy-self.__hls
            up = xy+self.__hls
            box = np.concatenate([lp,up], axis=1)
            return box

        if format == "matplotlib":
            lp = xy-self.__hls
            wl = 2*self.__hls
            box = np.concatenate([lp, wl], axis=1)
            return box

        raise NotImplementedError(f"Unknown formate: {format}")

    def intersection(self, xy:np.ndarray):
        boxes = self.get_box(xy, format="shapely")
        lp, up = boxes[:, :2], boxes[:, 2:]
        intersection_lp = np.maximum(lp[:, np.newaxis], lp[np.newaxis, ...])
        intersection_up = np.minimum(up[:, np.newaxis], up[np.newaxis, ...])
        _intersection = np.all(intersection_up > intersection_lp, axis=2)
        np.fill_diagonal(_intersection, False)
        result = np.zeros_like(_intersection, dtype=float)
        result[_intersection] = np.prod(intersection_up[_intersection]-intersection_lp[_intersection], axis=2)
        return result



