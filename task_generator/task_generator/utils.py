from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from rospkg import RosPack
import rospy
import os
import numpy as np
from nav_msgs.msg import OccupancyGrid

import heapq
import itertools

from task_generator.shared import Model, ModelType

class Utils:
    @staticmethod
    def get_simulator():
        return str(rospy.get_param("simulator", "flatland")).lower()
    
    @staticmethod
    def get_arena_type():
        return os.getenv("ARENA_TYPE", "training").lower()
    
    @staticmethod
    def generate_map_inner_border(free_space_indices, map_: OccupancyGrid):
        """generate map border (four vertices of the map)

        Returns:
            vertex_coordinate_x_y(np.ndarray with shape 4 x 2):
        """
        n_freespace_cells = len(free_space_indices[0])
        border_vertex=np.array([]).reshape(0, 2)
        border_vertices=np.array([]).reshape(0, 2)
        for idx in [0, n_freespace_cells-4]:
            y_in_cells, x_in_cells = free_space_indices[0][idx], free_space_indices[1][idx]
            y_in_meters = y_in_cells * map_.info.resolution + map_.info.origin.position.y
            x_in_meters = x_in_cells * map_.info.resolution + map_.info.origin.position.x
            border_vertex=np.vstack([border_vertex, [x_in_meters, y_in_meters]])
        border_vertices=np.vstack([border_vertices, [border_vertex[0,0],border_vertex[0,1]]])
        border_vertices=np.vstack([border_vertices, [border_vertex[0,0],border_vertex[1,1]]])
        border_vertices=np.vstack([border_vertices, [border_vertex[1,0],border_vertex[1,1]]])
        border_vertices=np.vstack([border_vertices, [border_vertex[1,0],border_vertex[0,1]]])
        # print('border',border_vertices)
        return border_vertices

    @staticmethod
    def update_freespace_indices_maze( map_: OccupancyGrid):
        """update the indices(represented in a tuple) of the freespace based on the map and the static polygons
        ostacles manuelly added 
        param map_ : original occupacy grid
        param vertlist: vertex of the polygons

        Returns:
            indices_y_x(tuple): indices of the non-occupied cells, the first element is the y-axis indices,
            the second element is the x-axis indices.
        """
        width_in_cell, height_in_cell = map_.info.width, map_.info.height
        map_2d = np.reshape(map_.data, (height_in_cell, width_in_cell))
        #height range and width range
        wall_occupancy=np.array([[1.25, 12.65, 10.6, 10.8],
                                                            [-4.45,18.35,16.3,16.5],
                                                            [-4.45, 18.35, 4.9, 5.1], 
                                                            [12.55, 12.75, -0.7, 22.1],
                                                            [1.15, 1.35, -0.7, 22.1],
                                                            [6.85, 7.05, 5.0, 16.4]])
        size=wall_occupancy.shape[0]
        for ranges in wall_occupancy:
            height_low = int(ranges[0]/map_.info.resolution)
            height_high = int(ranges[1]/map_.info.resolution)
            width_low = int(ranges[2]/map_.info.resolution)
            width_high = int(ranges[3]/map_.info.resolution)
            height_grid=height_high-height_low
            width_grid=width_high-width_low
            for i in range(height_grid):
                y =  height_low+ i
                for j in range(width_grid):
                    x= width_low + j
                    map_2d[y, x]=100
        free_space_indices_new = np.where(map_2d == 0)    
        return free_space_indices_new


class NamespaceIndexer:

    __freed: List[int]
    __gen: Iterator[int]
    __namespace: str
    __sep: str

    def __init__(self, namespace: str, sep: str = "_"):
        self.__freed = list()
        self.__gen = itertools.count()
        self.__namespace = namespace
        self.__sep = sep

    def free(self, index: int):
        heapq.heappush(self.__freed, index)

    def get(self) -> int:
        if len(self.__freed):
            return heapq.heappop(self.__freed)
        
        return next(self.__gen)
    
    def format(self, index: int) -> str:
        return f"{self.__namespace}{self.__sep}{index}"

    def __next__(self) -> Tuple[str, Callable[[], None]]:
        index = self.get()
        return self.format(index), lambda: self.free(index)


class ModelLoader:

    model_dir: str
    models: Iterable[str]
    __cache: Dict[str, Model]

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = [name for name, _, _ in os.walk(model_dir)]

    def load(self, model: str) -> Model:
        if model in self.__cache:
            return self.__cache[model]
        
        if model not in self.models:
            raise FileNotFoundError()
        
        with open(os.path.join(self.model_dir, model, "model.sdf")) as f:
            model_desc = f.read()
        
        model_obj = Model(
            type=ModelType.SDF,
            name=model,
            description=model_desc
        )
        return model_obj