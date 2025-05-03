# RDT for Kuavo robot

In this fork I use [.bag to .h5 transformer](https://github.com/b4r4d41z/rosbag2hdf5) for work with colectd data

Upload your `.hdf5` and `instruction_description.json` dataset fiels into data/Kuavo/ in format presented below 

```
/                        Group
/action                  Dataset {541, 20}
/base_action             Dataset {541, 2}
/instruction             Dataset {SCALAR}
/observations            Group
/observations/effort     Dataset {541, 14}
/observations/images     Group
/observations/images/cam_high Dataset {541, 480, 640, 3}
/observations/images/cam_left_wrist Dataset {541, 480, 640, 3}
/observations/images/cam_right_wrist Dataset {541, 480, 640, 3}
/observations/images_depth Group
/observations/images_depth/cam_high Dataset {541}
/observations/images_depth/cam_left_wrist Dataset {541}
/observations/images_depth/cam_right_wrist Dataset {541}
/observations/qpos       Dataset {541, 16}
/observations/qvel       Dataset {541, 14}
```