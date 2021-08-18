from typing import List

from BuildingBlocksGenetator import BuildingBlockHolder, BuildingBlock
from random import randint
import numpy as np
import utils
MAX_NUM_OF_TRIANGLES_INIT = 20


class BuildingBlockPopulation:

    @staticmethod
    def get_random_building_block():
        tmp_triangle = BuildingBlock()
        ang = randint(20, 90)
        edge1 = randint(1, 5)
        edge2 = randint(1, 5)
        tmp_triangle.generate_triangle(ang, edge1, edge2)
        return BuildingBlockHolder(tmp_triangle)


class GeneticBridge:

    def __init__(self):  # random init

        self.building_blocks_num = randint(10, MAX_NUM_OF_TRIANGLES_INIT)
        self.building_blocks = []
        self.ordered_building_blocks = []
        self.edges_pairs = List[tuple]
        self.order = list(np.random.permutation(self.building_blocks_num))

        for _ in range(self.building_blocks_num):
            self.building_blocks.append(BuildingBlockPopulation.get_random_building_block())

        self.order_by_permutation()
        self.order_edges_in_pairs()
        self.generate_coordinates()

    def order_edges_in_pairs(self):

        self.edges_pairs = [None] * (self.building_blocks_num - 1)
        for i in range(len(self.edges_pairs)):
            pair = self.get_legal_pairing(i)
            self.update_pair(pair, i, i)

    def order_by_permutation(self):

        self.ordered_building_blocks = [None] * self.building_blocks_num

        for idx, building_blk_idx in enumerate(self.order):
             self.ordered_building_blocks[idx] = self.building_blocks[
                 building_blk_idx] # TODO : copy the original triangle

    def get_legal_pairing(self, idx_of_1st_triangle):

        is_legal = False
        pair = tuple()

        while not is_legal:
            edge_idx1 = randint(0, 2)
            edge_idx2 = randint(0, 2)
            pair = (edge_idx1, edge_idx2)
            is_legal = self.is_legal_pairing(pair, idx_of_1st_triangle)

        return pair

    def is_legal_pairing(self, pair, idx_of_1st_triangle):

        edge_of_1st, edge_of_2nd = pair

        is_edge_taken1 = \
        self.ordered_building_blocks[idx_of_1st_triangle].is_edge_taken[
            edge_of_1st]
        is_edge_taken2 = self.ordered_building_blocks[idx_of_1st_triangle + 1
                                                      ].is_edge_taken[
            edge_of_2nd]

        return not is_edge_taken1 and not is_edge_taken2

    def update_pair(self, pair, idx_of_first_triangle, idx_of_pair):
        self.edges_pairs[idx_of_pair] = pair

        edge_of_1st, edge_of_2nd = pair

        self.ordered_building_blocks[idx_of_first_triangle].is_edge_taken[edge_of_1st]\
            = True

        self.ordered_building_blocks[idx_of_first_triangle + 1].is_edge_taken[
            edge_of_2nd] = True

    def generate_coordinates(self):

        prev_triangle = self.ordered_building_blocks[0]
        for i in range(1, len(self.ordered_building_blocks), 1):
            edge1, edge2 = self.edges_pairs[i - 1]
            self.ordered_building_blocks[i].align_to(prev_triangle, edge1,
                                                     edge2, i)

            prev_triangle = self.ordered_building_blocks[i]

    def plot(self):

        coors_list = [triangle.get_coors() for triangle in self.ordered_building_blocks]
        for i in range(1, len(coors_list)):
            utils.plot_triangle(coors_list[:i], title=f"with {i} triangles",
                                path=f"random_bridge/full_bridge_{i}")

    def get_fitness(self):
        pass