from task_generator.constants import Constants
import rospy
import numpy as np
import os
import xml.etree.ElementTree as ET
import rospkg
from task_generator.manager.pedsim_manager import PedsimManager
import time

# Define the ObstacleManager class
class ObstacleManager:
    def __init__(self, namespace, map_manager, simulator):
        # Initialize the ObstacleManager with required parameters
        self.map_manager = map_manager
        self.namespace = namespace
        self.simulator = simulator
        self.first_reset = True
        self.pedsimManager = PedsimManager(namespace)

    # TASK MODE SCENARIO
    def start_scenario(self, scenario):
        # Print a message indicating the start of spawning scenario obstacles
        print("spawning scenario obstacles")
        if rospy.get_param("pedsim"):
            # Spawn Pedsim map obstacles
            self.pedsimManager.spawn_pedsim_map_obstacles()
            # Spawn dynamic scenario obstacles
            self.pedsimManager.spawn_pedsim_dynamic_scenario_obstacles(scenario["obstacles"]["dynamic"])
            # Spawn static scenario obstacles with interaction_radius = 0.0 ( interactive obstalces are static obstacles with interaction_radius = 0.0)
            self.pedsimManager.spawn_pedsim_scenario_obstacles(scenario["obstacles"]["static"], interaction_radius=0.0)
            # Spawn interactive scenario obstacles with interaction_radius = 1.0
            self.pedsimManager.spawn_pedsim_scenario_obstacles(scenario["obstacles"]["interactive"], interaction_radius=1.0)

    def reset_scenario(self, scenario):
        if rospy.get_param("pedsim"):
            # Remove interactive and static obstacles in Pedsim
            self.pedsimManager.remove_interactive_obstacles_pedsim()
        # Remove all obstacles in the simulator
        self.simulator.remove_all_obstacles()

        if rospy.get_param("pedsim"):
            # Spawn dynamic scenario obstacles again after reset
            self.pedsimManager.spawn_pedsim_dynamic_scenario_obstacles(scenario["obstacles"]["dynamic"])
            # Spawn static scenario obstacles again after reset with interaction_radius = 0.0
            self.pedsimManager.spawn_pedsim_scenario_obstacles(scenario["obstacles"]["static"], interaction_radius=0.0)
            # Spawn interactive scenario obstacles again after reset with interaction_radius = 1.0
            self.pedsimManager.spawn_pedsim_scenario_obstacles(scenario["obstacles"]["interactive"], interaction_radius=1.0)

    # TASK MODE RANDOM
    def reset_random(
            self, 
            dynamic_obstacles=Constants.ObstacleManager.DYNAMIC_OBSTACLES,
            static_obstacles=Constants.ObstacleManager.STATIC_OBSTACLES,
            interactive_obstacles=Constants.ObstacleManager.INTERACTIVE_OBSTACLES,
            forbidden_zones=[]
        ):

        if forbidden_zones is None:
            forbidden_zones = []

        if self.first_reset:
            self.first_reset = False
        else:  
            # Remove all obstacles in the simulator
            self.simulator.remove_all_obstacles() 
            if rospy.get_param("pedsim"):
                # Remove interactive obstacles in Pedsim
                self.pedsimManager.remove_interactive_obstacles_pedsim()

        # Initialize arrays for dynamic, static, and interactive obstacles
        dynamic_obstacles_array = np.array([],dtype=object).reshape(0,3)
        static_obstacles_array = np.array([],dtype=object).reshape(0,2)
        interactive_obstacles_array = np.array([],dtype=object).reshape(0,2)

        if rospy.get_param("pedsim"):
            # Add Pedsim map obstacles to forbidden zones
            forbidden_zones = forbidden_zones + self.pedsimManager.spawn_pedsim_map_obstacles()

        # Create static obstacles
        for i in range(static_obstacles):
            if rospy.get_param("pedsim"):
                # Create a static Pedsim obstacle
                x = self.pedsimManager.create_pedsim_obstacle(False, i ,self.map_manager, forbidden_zones)
                forbidden_zones.append([x[1][0], x[1][1], 40])
                static_obstacles_array = np.vstack((static_obstacles_array, x))
            else: 
                pass

        if static_obstacles_array.size > 0:
            # Spawn static Pedsim obstacles with interaction_radius = 0.0
            self.pedsimManager.spawn_pedsim_obstacles(static_obstacles_array, interaction_radius=0.0)

        # Create interactive obstacles  
        for i in range(interactive_obstacles):
            if rospy.get_param("pedsim"):
                # Create an interactive Pedsim obstacle
                x = self.pedsimManager.create_pedsim_obstacle(False, i,self.map_manager, forbidden_zones)
                forbidden_zones.append([x[1][0], x[1][1], 40])
                interactive_obstacles_array = np.vstack((interactive_obstacles_array, x))
            else: 
                pass

        if interactive_obstacles_array.size > 0:
            # Spawn interactive Pedsim obstacles with interaction_radius = 1.0
            self.pedsimManager.spawn_pedsim_obstacles(interactive_obstacles_array, interaction_radius=1.0)

        # Create dynamic obstacles 
        for i in range(dynamic_obstacles):
            if rospy.get_param("pedsim"):
                # Create a dynamic Pedsim obstacle
                x = self.pedsimManager.create_pedsim_obstacle(True, i,self.map_manager, forbidden_zones)
                dynamic_obstacles_array = np.vstack((dynamic_obstacles_array, x))
            else: 
                pass

        if dynamic_obstacles_array.size > 0:
            # Spawn dynamic Pedsim obstacles
            self.pedsimManager.spawn_pedsim_dynamic_obstacles(dynamic_obstacles_array)

    # TASK MODE RANDOM SCENARIO
    def reset_random_scenario(
            self, 
            dynamic_obstacles=Constants.ObstacleManager.DYNAMIC_OBSTACLES,
            static_obstacles=Constants.ObstacleManager.STATIC_OBSTACLES,
            interactive_obstacles=Constants.ObstacleManager.INTERACTIVE_OBSTACLES,
            forbidden_zones=[]
        ):

        if forbidden_zones is None:
            forbidden_zones = []

        if self.first_reset:
            self.first_reset = False
        else:  
            # Remove all obstacles in the simulator
            self.simulator.remove_all_obstacles()
            if rospy.get_param("pedsim"):
                # Remove interactive and static obstacles in Pedsim
                self.pedsimManager.remove_interactive_obstacles_pedsim()
        
        # Define the path to the XML config file
        xml_path = os.path.join(
        rospkg.RosPack().get_path("task_generator"), 
        "scenarios", 
        "random_scenario.xml")
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Extract the number of tables, shelves, adults, elders, and children from the XML
        num_tables = [int(root[0][0].text),root[0][1].text,root[0][2].text]
        num_shelves = [int(root[1][0].text),root[1][1].text,root[1][2].text]
        num_adults = [int(root[2][0].text),root[2][1].text,root[2][2].text]
        num_elder = [int(root[3][0].text),root[3][1].text,root[3][2].text]
        num_child = [int(root[4][0].text),root[4][1].text,root[4][2].text]

        dynamic_obstacles_array = np.array([],dtype=object).reshape(0,3)
        static_obstacles_array = np.array([],dtype=object).reshape(0,2)
        interactive_obstacles_array = np.array([],dtype=object).reshape(0,2)

        if rospy.get_param("pedsim"):
            # Add Pedsim map obstacles to forbidden zones 
            forbidden_zones = forbidden_zones + self.pedsimManager.spawn_pedsim_map_obstacles()

        # Create static obstacles from table model
        for ob_type in [num_tables]:
            static_obstacles_array = np.array([],dtype=object).reshape(0,2)
            for i in range(ob_type[0]):
                if rospy.get_param("pedsim"):
                    x = self.pedsimManager.create_pedsim_obstacle(False, i,self.map_manager, forbidden_zones)
                    forbidden_zones.append([x[1][0], x[1][1], 40])
                    static_obstacles_array = np.vstack((static_obstacles_array, x))
                else: 
                    pass

            if static_obstacles_array.size > 0:
                # Spawn static Pedsim obstacles for tables with interaction_radius = ob_type[1]
                self.pedsimManager.spawn_pedsim_obstacles(static_obstacles_array, ob_type[1], ob_type[2], interaction_radius=0.0)

        # Create interactive obstacles from shelves model
        for ob_type in [num_shelves]:
            interactive_obstacles_array = np.array([],dtype=object).reshape(0,2)
            for i in range(ob_type[0]):
                if rospy.get_param("pedsim"):
                    x = self.pedsimManager.create_pedsim_obstacle(False, i,self.map_manager, forbidden_zones)
                    forbidden_zones.append([x[1][0], x[1][1], 40])
                    interactive_obstacles_array = np.vstack((interactive_obstacles_array, x))
                else: 
                    pass

            if interactive_obstacles_array.size > 0:
                # Spawn interactive Pedsim obstacles for shelves with interaction_radius = ob_type[1]
                self.pedsimManager.spawn_pedsim_obstacles(interactive_obstacles_array, ob_type[1], ob_type[2], interaction_radius=1.0)

        # Create dynamic obstacles from adults, elders, and children models
        for ob_type in [num_adults, num_elder, num_child]:
            dynamic_obstacles_array = np.array([],dtype=object).reshape(0,3)
            for i in range(ob_type[0]):
                if rospy.get_param("pedsim"):
                    x = self.pedsimManager.create_pedsim_obstacle(True, i,self.map_manager, forbidden_zones)
                    dynamic_obstacles_array = np.vstack((dynamic_obstacles_array, x))
                else: 
                    pass

            if dynamic_obstacles_array.size > 0:
                # Spawn dynamic Pedsim obstacles for adults, elders, and children
                self.pedsimManager.spawn_pedsim_dynamic_obstacles(dynamic_obstacles_array, ob_type[1], ob_type[2])
