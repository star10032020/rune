from setuptools import find_packages, setup

package_name = "rune"

setup(
    name=package_name,
    version="0.0.0",
    #packages=find_packages(exclude=["test"]),
    packages=[package_name,package_name+"/module"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "rosidl_generator_py",
        "numpy==1.26.1",
        "typing_extensions>=4.8.0",
        "opencv-python",
        "ultralytics",
        "tensorrt",
    ],
    zip_safe=True,
    maintainer="ros",
    maintainer_email="1940574515@qq.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["RuneFinder = rune.RuneFinder:main",
         'testImageSender = rune.testImageSender:main',
            'testTopicGetter = rune.testTopicGetter:main',
        ],
        
    },
)
