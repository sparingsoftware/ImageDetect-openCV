
import cv2
import sys
import numpy
import glob
import os
import time




#
# USE: python3 detect_targets.py img-dir -dir_out img-out -targets targets.xml -kps_max_start 2000 -kps_max 200
#
# -dir_out              - folder with images with drawn keypoints
# -targets              - XML file for targets
# -kps_max_start        - number of keypoints to detect
# -kps_max              - number of final keypoints after ANMS
#
#



# c++: http://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/
# papers: http://matthewalunbrown.com/papers/cvpr05.pdf
def nonmaxSuppressionKeypoints( kps, minPoints ):
    kpsArray = numpy.array(kps)
    
    # if not enought
    if kpsArray.size < minPoints:
        return kps
    
    # sort by response (value of KeyPoint)
    kpsArray = sorted(kpsArray, key=lambda x: x.response, reverse=True)

    kpsArray = numpy.array(kpsArray)

    anmsKps = []
    radii = []
    radiiSorted = []

    # see paper
    robustCoeff = 1.11

    for i in range(0, kpsArray.size):
        response = kpsArray[i].response * robustCoeff
        radius = sys.float_info.max
        for j in range(0, i):

            # dist between points
            dist = cv2.norm( (kpsArray[i].pt[0] - kpsArray[j].pt[0], kpsArray[i].pt[1] - kpsArray[j].pt[1]) )
            radius = min( radius, dist )
            
            if kpsArray[j].response <= response:
                break

        radii.append(radius)
        radiiSorted.append(radius)

    radiiSorted = sorted(radiiSorted, reverse=True)
    radii = numpy.array(radii)

    # get last acceptable radius
    decisionRadius = radiiSorted[minPoints]

    # get best keypoints
    for i in range(0, radii.size):
        if radii[i] >= decisionRadius:
            anmsKps.append(kpsArray[i])

    return anmsKps


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


# get arguments
if len(sys.argv) < 2:
    print("ERROR: Provide images dir")
    sys.exit()

imgDir = sys.argv[1]


imgOutDir = ""
targetsFile = ""
maxStartKps = 2000
maxKps = 200

# get params
for i in range(1, len(sys.argv)-1):
    arg = sys.argv[i]
    if arg == "-dir_out":
        imgOutDir = sys.argv[i+1]
        print("[PARAM] DIR OUT : %s" % imgOutDir)

    if arg == "-targets":
        targetsFile = sys.argv[i+1]
        print("[PARAM] TARGETS : %s" % targetsFile)

    if arg == "-kps_max_start":
        maxStartKps = int(sys.argv[i+1])
        print("[PARAM] MAX START KEYPOINTS : %d" % maxStartKps)

    if arg == "-kps_max":
        maxKps = int(sys.argv[i+1])
        print("[PARAM] MAX KEYPOINTS : %d" % maxKps)



# create out dir if not exists
if imgOutDir != "":
    if not os.path.exists(imgOutDir):
        os.makedirs(imgOutDir)

# read all images fro dir
types = (imgDir + '/*.jpg', imgDir + '/*.JPG', imgDir + '/*.png', imgDir + '/*.PNG') # the tuple of file types
images = []
for files in types:
    imgs = glob.glob(files)
    for imageFile in imgs:
        image = os.path.basename(imageFile)
        images.append(image)

print("IMAGES DETECTED = %d" % len(images))

#  create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);
# Initiate STAR detector
orb = cv2.ORB_create()
orb.setMaxFeatures(maxStartKps)
#orb.setFastThreshold(5)
#orb.setEdgeThreshold(5)
#orb.setPatchSize(20)

#orb.setScaleFactor(1.2)
#orb.setNLevels(4)

# save to file
if targetsFile != "":
    cv_file = cv2.FileStorage(targetsFile, cv2.FILE_STORAGE_WRITE)


ids = []

for i in range(0, len(images)):
    print("Calculating [%s]: %d/%d" % (images[i], i, len(images)))
    
    img = cv2.imread(imgDir + "/" + images[i])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find the keypoints with ORB
    keypoints = orb.detect(img, None)

    # ANMS filter keypoints
    keypoints = nonmaxSuppressionKeypoints(keypoints, maxKps)

    # compute the descriptors with ORB
    keypoints, des = orb.compute(gray, keypoints)
    
    # drawing keypoints
    if imgOutDir != "":
        # draw keypoints
        for point in keypoints:
            cv2.circle(gray, (int(point.pt[0]), int(point.pt[1])), 3, (255, 0, 0), -1)
        
        cv2.imwrite(imgOutDir + "/" + images[i], gray)

    # targets XML file
    if targetsFile != "":
        # convert keypints to array
        # https://stackoverflow.com/questions/25680529/store-the-extracted-surf-descriptors-and-keypoints-in-npy-file
        kpMat = numpy.array([[kp.pt[0], kp.pt[1], kp.size,
                              kp.angle, kp.response, kp.octave,
                              kp.class_id]
                              for kp in keypoints])
        
        imgId = images[i].split(".")[0]
        ids.append(int(imgId))
        cv_file.write(("descriptors_%s" % imgId), des)
        cv_file.write(("keypoints_%s" % imgId), kpMat)


#
cv_file.write("ids", numpy.array(ids))

# note you *release* you don't close() a FileStorage object
cv_file.release()


