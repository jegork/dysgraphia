import datetime
from dataclasses import dataclass
from typing import Any, List

import geopandas as gpd
import movingpandas as mpd
import numpy as np
import pandas as pd
from pyproj import CRS
from rdp import rdp
from shapely.geometry import Point

from bezier import *

now = datetime.datetime(year=2021, month=1, day=1)


def angle(directions):
    """Return the angle between vectors
    """
    vec2 = np.round(directions[1:], 10)
    vec1 = np.round(directions[:-1], 10)

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))

    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
    return np.arccos(np.round(cos, 10))


@dataclass
class Writing:
    data: Any
    label: int
    type_: int
    user_id: int

    def split(self) -> List[np.ndarray]:
        """Split data using Ramer-Douglas-Peucker algorithm only.

        Returns:
            List[np.ndarray]: List of splits.
        """
        coords = self.data[:, 0:2]
        times = self.data[:, 2]
        mask = rdp(coords, return_mask=True)

        times = times[mask]
        coords = coords[mask]

        min_angle = np.pi / 2.5
        directions = np.diff(coords, axis=0)
        theta = angle(directions)

        idx = np.where(theta > min_angle)[0] + 1

        sx, sy = coords.T

        return [arr for arr in np.split(np.asarray([sx, sy, times]).T, idx) if len(arr.shape) > 1]

    def split_with_mpd(self) -> List[np.ndarray]:
        """Split data using movingpandas SpeedSplitter and Ramer-Douglas-Peucker algorithm afterwards.

        Returns:
            List[np.ndarray]: List of splits.
        """
        coords = self.data[:, 0:2]
        time = [now + datetime.timedelta(milliseconds=t) for t in self.data[:, 2]]

        df = pd.DataFrame({
            'geometry': [Point(x, y) for x, y in coords],
            't': time,
        }).set_index('t')
        gdf = gpd.GeoDataFrame(df, crs=CRS(31256))
        mdf = mpd.Trajectory(gdf, 1)

        col = mpd.SpeedSplitter(mdf).split(speed=1, duration=datetime.timedelta(milliseconds=100))

        trajectories = []
        for t in col.trajectories:
            trajectory = [(point.coords[0][0], point.coords[0][1]) for point in
                          list(t.to_point_gdf()['geometry'].values)]
            trajectory = np.asarray(trajectory)

            mask = rdp(trajectory, return_mask=True)
            simplified_trajectory = trajectory[mask]

            times = [int((t - now).to_pytimedelta().total_seconds() * 1000) for t in t.to_point_gdf().index]
            times = np.asarray(times)
            times = times[mask]

            sx, sy = simplified_trajectory.T

            min_angle = np.pi / 2.5

            # Compute the direction vectors on the simplified_trajectory.
            directions = np.diff(simplified_trajectory, axis=0)
            theta = angle(directions)

            # Select the index of the points with the greatest theta.
            # Large theta is associated with greatest change in direction.
            idx = np.where(theta > min_angle)[0] + 1

            sub_trajectories = [arr for arr in np.split(np.asarray([sx, sy, times]).T, idx) if len(arr.shape) > 1]
            trajectories.extend(sub_trajectories)

        return trajectories

    def get_bezier_features(self, use_mpd: bool = True) -> None:
        """Get features (introduced in https://arxiv.org/pdf/1902.10525.pdf) based on Bezier curve interpolation.
        """
        def get_data(t):
            def calc(a, b, c, d):
                vector = d - a

                dist_1 = np.linalg.norm(b - a) ** 2
                dist_2 = np.linalg.norm(c - d) ** 2

                angle_1 = get_angle(b, a)
                angle_2 = get_angle(c, d)

                return np.asarray([vector[0], vector[1], dist_1, dist_2, angle_1, angle_2])

            params = get_bezier_parameters(t[:, 0], t[:, 1], degree=3)
            params = [np.asarray(p) for p in params]

            return calc(*params)

        trajectories = self.split_with_mpd() if use_mpd else self.split()

        _t = [np.append(get_data(traj[:, 0:2]), traj[0, 2]) for traj in trajectories if traj.shape[0] > 3]
        # for idx in range(len(trajectories)):

        if len(_t) != 0:
            self.bezier_features = np.vstack(_t)
        else:
            self.bezier_features = None
