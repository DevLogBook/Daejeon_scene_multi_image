import torch

from dense_match.network import AgriMatcher
from warp.filter import robust_homography_estimation

checkpoint = ""

model = AgriMatcher(128, 256, 32)

def solving_h(img1, img2):
    
    pass