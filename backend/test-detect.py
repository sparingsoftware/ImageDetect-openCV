import cv2
import numpy
import json
import sys
import time



MIN_MATCHES = 4

IS_DEV = True

def lambda_handler(event, context):
    
    start_time = time.time()

    targetsFile = "targets300.xml"
    
    # load targets
    cv_file = cv2.FileStorage(targetsFile, cv2.FILE_STORAGE_READ)
    ids = cv_file.getNode("ids").mat()

    # find best detect
    detected = []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    for frame in event['frames']:
        # create descriptor of scene
        rows = frame['rows']
        cols = frame['cols']
        dsc = frame['descriptors']
        kps = frame['keypoints']
        
        sceneDsc = numpy.zeros((rows, cols), dtype='uint8')
        
        for i in range(0, len(dsc)):
            row = int(i/cols)
            col = int(i%cols)
            sceneDsc[row][col] = dsc[i]

        for id in ids:
            dscName = ("descriptors_%s" % id[0])
            targetDsc = cv_file.getNode(dscName).mat()
            
            kpsName = ("keypoints_%s" % id[0])
            targetKps = cv_file.getNode(kpsName).mat()
            
            # target and scene may have different number of descriptors
            maxDsc = min(len(sceneDsc), len(targetDsc))
            
            # Match descriptors.
            matches = matcher.knnMatch(sceneDsc, targetDsc, k=2)
            goodMatches = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    # Dont know why but sometimes indexes are exceeded
                    if m.queryIdx < len(targetKps) and m.trainIdx < len(kps):
                        goodMatches.append(m)
        
            matchesCount = len(goodMatches)
            if matchesCount > MIN_MATCHES:
                id_ = int(id[0])
                isNew = True

#if IS_DEV == True:
                    #print("id = %d, matchesCount = %d, size = %d" % (id_, matchesCount, len(targetKps)))
                    
                    #print("targetKps = %d, kps = %d" % (len(targetKps), len(kps)))

#for m in goodMatches:
                #print('queryIdx = %d, trainIdx = %d' % (m.queryIdx, m.trainIdx))

                # check homography
                targetH = numpy.float32([ [targetKps[m.queryIdx][0], targetKps[m.queryIdx][1]] for m in goodMatches ]).reshape(-1,1,2)

                sceneH = numpy.float32([ kps[m.trainIdx] for m in goodMatches ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(targetH, sceneH, cv2.RANSAC)

#if IS_DEV == True:
#print(M)

                # go futher only if homography is OK
                if type(M) is numpy.ndarray:
                    # if already in detected array, just add matches
                    for det in detected:
                        if det['id'] == id_:
                            isNew = False
                            det['matches'] += matchesCount
                            det['frames'] += 1
                
                    if isNew == True:
                        detected.append({'id': id_, 'matches': matchesCount, 'frames': 1})


    executionTime = time.time() - start_time

    # get beer data
    
    beers = json.load(open('beers.json'))
    for detected_ in detected:
        for beer in beers:
            if detected_['id'] == beer['bid']:
                detected_['beer_name'] = beer['beer_name']
                detected_['rating_score'] = beer['rating_score']
                detected_['rating_count'] = beer['rating_count']
                detected_['beer_ibu'] = beer['beer_ibu']

    return json.dumps({'results': detected, 'execution_time': executionTime})


######

if IS_DEV == True:
    targetFile = sys.argv[1]

    target = json.load(open(targetFile))

    print(lambda_handler(target, 42))
