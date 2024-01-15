cmake_minimum_required(VERSION 3.5)
project(rune)


# 默认到C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(msg_pkg REQUIRED)

set(msg_files
"msg/RuneCorner.msg"
)
message(${PROJECT_NAME})
rosidl_generate_interfaces(${PROJECT_NAME}
${msg_files}
DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)
# 安装Python模块
ament_python_install_package(${PROJECT_NAME})

# 安装启动文件和其他资源，例如配置文件
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)
#ament_export_dependencies(rosidl_default_runtime)
ament_package()
