cmake_minimum_required(VERSION 2.8.3)
project(hmcl_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs  
  message_generation
  sensor_msgs
  nav_msgs
  std_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
  VehiclePredictionROS.msg
    obstacle.msg
    Track.msg
    TrackArray.msg
    Trafficlight.msg
    BehaviorFactor.msg
    TransitionCondition.msg
    DTLane.msg
    Lane.msg
    LaneArray.msg
    Waypoint.msg    
    WaypointArray.msg    
    WaypointState.msg    
    MissionWaypoint.msg    
    Waypointhmcl.msg 
    RemoteButton.msg
    VehicleGear.msg
    VehicleLight.msg
    VehicleSCC.msg
    VehicleStatus.msg
    VehicleSteering.msg
    VehicleWheelSpeed.msg
    CtrlVehicleStates.msg    
)

add_service_files(
  FILES RecognizeLightState.srv mpcc.srv
)

generate_messages(
  DEPENDENCIES
    geometry_msgs    
    sensor_msgs
    nav_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs    
    message_runtime
    sensor_msgs
    std_msgs
    nav_msgs
)
