# ur5-robotic-grasping
This repository implements the grasp inference method of Kumra et al. (2020) in a robotic simulation developed in Pybullet. Kumra et al. (2020) propose a generative residual convolutional neural network based model architecture which predicts a suitable antipodal grasp for objects using an image of the object scene. Three different grasping scenarios have been implemented. These include objects in isolation, objects packed together, and objects in a pile (Kasaei et al., 2021). 

All code in the directory 'network' is an adaptation of Kumra's open source code that was taken from the following repository: https://github.com/skumra/robotic-grasping  
The simulation code is an adaptation from the following repository: https://github.com/ElectronicElephant/pybullet_ur5_robotiq  
Object models were taken from the following repository: https://github.com/eleramp/pybullet-object-models

## Requirements
- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow
- pandas
- matplotlib
- pybullet

## Demo
Running the script 'demo.py' gives a demonstration of the simulation. The demo can be run with three different grasping scenarios. Run 'demo.py --help' to see a full list of options.

Example:
```bash
python demo.py --scenario=isolated --runs=1 --show-network-output=False
```
## References
Sulabh Kumra, Shirin Joshi, and Ferat Sahin.  Antipo-dal robotic grasping using generative residual convo-lutional neural network. In2020 IEEE/RSJ Interna-tional Conference on Intelligent Robots and Systems(IROS), pages 9626–9633, 2020.  doi: 10.1109/IROS45743.2020.9340777.  
Hamidreza  Kasaei  and  Mohammadreza  Kasaei.   MV-grasp:   Real-time   multi-view   3D   object   graspingin  highly  cluttered  environments.arXiv preprintarXiv:2103.10997, 2021.