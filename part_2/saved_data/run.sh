#!/bin/bash
# cd ..

# python3 evo_helper.py ./saved_data/edges-poses.g2o ./saved_data/edges-poses.kitti
# python3 evo_helper.py  ./saved_data/opt-poses.g2o ./saved_data/opt-poses.kitti

# evo_ape kitti gt.kitti opt-poses.kitti -v --plot --plot_mode xy
evo_rpe kitti gt.kitti edges-poses.kitti -v --plot --plot_mode xy