cmake_minimum_required(VERSION 3.0.2)
project(racepkg)

find_package(catkin REQUIRED COMPONENTS
  hmcl_msgs
  nav_msgs
  roscpp
  rospy  
  dynamic_reconfigure
)


catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/racepkgDyn.cfg
  #...
)

catkin_package(

)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
