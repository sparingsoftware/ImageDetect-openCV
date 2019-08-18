# ImageDetect-openCV
Samples for detecting images on iOS/Android with OpenCV


## Description

This is not a final solution - it is a collection of scripts from few projects which can be usefull for someone who would like to dive deep into OpenCV on mobile.

Repo is divided into 3 modules:
1. detect targets - it is a Python script which extracts keypoints and their descriptors from images and saves it into XML file 

![Image](image_features.png)

2. backend - simple script that I used on AWS Lambda for matching images on backend
3. iOS - iOS Objective-c / C++ modules which enables to load XML with keypoints and find matches as:
``` swift
val targetDetector = TargetDetector()
targetDetector.loadTargets("path to xml")
let results = targetDetector.detectTargets(image)
```

## License
MIT License © [Sparing Interactive](https://github.com/SparingSoftware)
