rgb_path=YcbMediumClamp1.png
depth_path=YcbMediumClamp1.tiff

python run_offline.py --network=trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97 --rgb_path=test_input2/$rgb_path --depth_path=test_input2/$depth_path