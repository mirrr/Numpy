import dumbpy as dp
import numc as nc
import numpy as np
import hashlib, struct
from typing import Union, List
import operator
import time

def test_small_add():
    # TODO: YOUR CODE HERE
    dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
    dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
    is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
    self.assertTrue(is_correct)
    print_speedup(speed_up)

test_small_add()
