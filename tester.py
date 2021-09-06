import BuildingBlocksGenetator
import Bridge
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from utils import *
import random

OUTPUT_PATH = r"C:\Users\orger\Desktop\Studies\Year_3\SemesterB\genetics" \
              r"\final_project\py_project"


tmp_triangle = BuildingBlocksGenetator.BuildingBlock()
tmp_triangle.generate_triangle(90, 4, 3)

triangle2 = BuildingBlocksGenetator.BuildingBlock()
triangle2.generate_triangle(70, 4, 6)

triangle1t = BuildingBlocksGenetator.BuildingBlockHolder(tmp_triangle)
triangle2t = BuildingBlocksGenetator.BuildingBlockHolder(triangle2)

# plot_triangle([triangle1t.get_coors()])
# plot_triangle([triangle2t.get_coors()])

coors_to_plot = []
edges_idx_pairs = []


# for i in range(3):
#     for k in range(3):
#         triangle2t.align_to(triangle1t, i, k)
#         coors2_after_alignment = triangle2t.get_coors()
#         coors_to_plot.append([base_coors, coors2_after_alignment])
#         edges_idx_pairs.append((i, k))


#plot_N_times(coors_to_plot, edges_idx_pairs)

# base_coors = triangle1t.get_coors()
# coors_plot_series = [base_coors]
# prev_triangle = triangle1t
#
# for _ in range(10):
#
#     tmp_triangle = BuildingBlocksGenetator.BuildingBlock()
#     ang = random.randint(20, 90)
#     edge1 = random.randint(1, 5)
#     edge2 = random.randint(1, 5)
#     tmp_triangle.generate_triangle(ang, edge1, edge2)
#     tmp_triangle_t = BuildingBlocksGenetator.BuildingBlockHolder(
#         tmp_triangle)
#
#     tmp_triangle_t.align_to(prev_triangle, 1, 0)
#     coors_plot_series.append(tmp_triangle_t.get_coors())
#     prev_triangle = tmp_triangle_t
#     plot_triangle(coors_plot_series)


from EvolutionRunner import run_manager, BuildingBlockPopulation, pairs_mutation


# bridge = Bridge.GeneticBridge(BuildingBlockPopulation(30))
# bridge.plot_growth()
# bridge.change_bridge_size(bridge.size + 3)

# run_manager()

bridge = Bridge.GeneticBridge(BuildingBlockPopulation(30))
bridge.plot_with_target(title="Before mutation")
bridge = pairs_mutation(bridge, 1.1)
bridge.plot_with_target(title="Pairs mutation")



# long_edges = []
# edges_b = []
# for i in range(360):
#     long_edge = triangle1.get_polygon(0, 0, i)
#     long_edges.append(long_edge)
#
#
# bins = np.unique(long_edges)
# bins_num_a = len(np.unique(long_edges))
# plt.hist(long_edges, bins=bins_num_a)
# plt.show()
#
# bins_num_b = len(np.unique(edges_b))
# plt.hist(edges_b, bins=bins_num_b)
# plt.show()

