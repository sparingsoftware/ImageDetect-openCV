//
//  TargetDetector.m
//  P1V0
//
//  Created by Ufos on 29.11.2017.
//  Copyright © 2017 Panowie Programisci. All rights reserved.
//

// ORDER IS IMPORTANT! openCV must be imported BEFORE ukit because of some conflicts
#import <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video/tracking.hpp>

#import "TargetDetector.h"

using namespace cv;
using namespace std;

//

int MIN_MATCHES_TO_DETECT = 4;
int IMAGE_SIZE = 400;


int MAX_KP = 2000;
int MAX_KP_FILTER = 500;

int MAX_KP_FILTER_CONFIRM= 500;

//

@interface TargetDetector ()

@property vector<vector<KeyPoint>> *targetKeypoints;
@property vector<Mat> *targetDescriptors;
@property vector<int> *targetIds;

@end

//

@implementation TargetDetector


/////


- (instancetype)init {
    if (self = [super init]) {
        _targetKeypoints = new vector<vector<KeyPoint>>();
        _targetIds = new vector<int>();
        _targetDescriptors = new vector<Mat>();
    }
    
    return self;
}

- (void)dealloc {
    delete _targetKeypoints;
    delete _targetIds;
    delete _targetDescriptors;
}


/////

//

- (BOOL) addTarget: (UIImage*)image {
    Mat targetMat = [self cvMatFromUIImage:image];
    
    // grey scale
    cvtColor(targetMat, targetMat, CV_BGR2GRAY);
    
    // filters for corners
    // https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    /*
    Mat imgSmoothed;
    normalize(targetMat, targetMat, 0, 255, NORM_MINMAX);
    GaussianBlur(targetMat, imgSmoothed, cv::Size(0, 0), 3);
    addWeighted(targetMat, 1.5, imgSmoothed, -0.5, 0, targetMat);
    */
    // create ORB feature/descriptor detector
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(MAX_KP);
    
    std::vector<cv::KeyPoint> kp;
    cv::Mat dsc;
    
    orb->detectAndCompute(targetMat, cv::noArray(), kp, dsc);
    
    if (!dsc.isContinuous()) {
        return false;
    }
    
    // add to vector
    [self targetKeypoints]->push_back(kp);
    [self targetDescriptors]->push_back(dsc);
    
    return true;
}

- (void) loadTargets: (NSString*)filename {
    
    NSString* path = [[NSBundle mainBundle] pathForResource:filename ofType:@"xml"];
    
    FileStorage fs([path UTF8String], FileStorage::READ);
    
    cv::Mat ids;
    
    // read all ids in target file
    fs["ids"] >> ids;

    // read all descriptors
    for (int i=0; i<ids.rows; i++) {
        cv::Mat dsc;
        std::vector<cv::KeyPoint> kp;
        
        int targetId = ids.at<int>(i, 0);
        
        NSString *dscKey = [NSString stringWithFormat:@"descriptors_%d", targetId];
        NSString *kpsKey = [NSString stringWithFormat:@"keypoints_%d", targetId];
        
        fs[dscKey.UTF8String] >> dsc;
        fs[kpsKey.UTF8String] >> kp;
        
        // add descriptors to target list
        [self targetKeypoints]->push_back(kp);
        [self targetDescriptors]->push_back(dsc);
        [self targetIds]->push_back(targetId);
    }
    
    fs.release();
}


// TODO: 1) function getImageKpAndDsc 2) findResultOnScene(mat, for object: id)

- (NSArray<DetectResults*>*) detectTargets: (UIImage*)scene {
    NSMutableArray<DetectResults*> *results = [NSMutableArray<DetectResults*> new];
    
    // detect keypoints and descriptors
    Mat sceneMat = [self cvMatFromUIImage:scene];
    
    std::vector<cv::KeyPoint> sceneKp;
    cv::Mat sceneDsc;
    
    findFeatures(sceneMat, sceneKp, sceneDsc, MAX_KP, MAX_KP_FILTER);
    
    if (!sceneDsc.isContinuous()) {
        return results;
    }
    
    // find matches
    FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
    
    for (int n = 0; n<[self targetDescriptors]->size(); n++) {
        std::vector<cv::KeyPoint> kp = [self targetKeypoints]->at(n);
        cv::Mat dsc = [self targetDescriptors]->at(n);

        // filter for best matches
        std::vector<DMatch> good_matches;
        findMatches(&matcher, dsc, sceneDsc, good_matches);
        
        int matchesCount = int(good_matches.size());
        
        // we got a match (probably)
        if (matchesCount >= MIN_MATCHES_TO_DETECT) {
            int targetId = [self targetIds]->at(n);
            DetectResults *res = [[DetectResults alloc] initWithMatches:matchesCount forObject:targetId withBounds:CGRectZero];
            [results addObject:res];
            
            /*
            CGRect bounds = [self findHomography:good_matches withTarget:kp onScene:sceneKp];
            int targetId = [self targetIds]->at(n);
            
            if (bounds.size.width > 0) {
                DetectResults *res = [[DetectResults alloc] initWithMatches:matchesCount forObject:targetId withBounds:bounds];
                [results addObject:res];
            }*/
        }
    }
    return results;
}


//


- (NSArray<DetectResults*>*) detectTargetWithIds: (NSArray*)objecstId onScene: (UIImage*)scene {
    
    NSMutableArray<DetectResults*> *results = [NSMutableArray<DetectResults*> new];
    
    // detect keypoints and descriptors
    Mat sceneMat = [self cvMatFromUIImage:scene];
    
    std::vector<cv::KeyPoint> sceneKp;
    cv::Mat sceneDsc;
    
    findFeatures(sceneMat, sceneKp, sceneDsc, MAX_KP, MAX_KP_FILTER_CONFIRM);
    
    if (!sceneDsc.isContinuous()) {
        return results;
    }
    
    
    // find matches
    FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
    
    // BFMatcher bruteMatcher(cv::NORM_HAMMING);
    
    for(int i=0; i<objecstId.count; i++) {
        int objectId = [[objecstId objectAtIndex:i] intValue];
        
        // get infex num
        int n = 0;
        for (int i = 0; i<[self targetDescriptors]->size(); i++) {
            int targetId = [self targetIds]->at(i);
            if (targetId == objectId) {
                n = i;
            }
        }
        
        // get target's data
        std::vector<cv::KeyPoint> kp = [self targetKeypoints]->at(n);
        cv::Mat dsc = [self targetDescriptors]->at(n);
        
        // MATCH filter for best matches
        std::vector<DMatch> good_matches;
        findMatches(&matcher, dsc, sceneDsc, good_matches);
        
        int matchesCount = int(good_matches.size());
        
        // we got a match (probably)
        if (matchesCount >= MIN_MATCHES_TO_DETECT) {
            
            CGRect bounds = [self findHomography:good_matches withTarget:kp onScene:sceneKp];
            int targetId = [self targetIds]->at(n);
            
            if (bounds.size.width > 0) {
                DetectResults *res = [[DetectResults alloc] initWithMatches:matchesCount forObject:targetId withBounds:bounds];
                [results addObject:res];
            }
        }
    }
    
    return results;
}


//


- (DetectResults*) detectTargetWithId: (int)objectId onScene: (UIImage*)scene {
    
    // detect keypoints and descriptors
    Mat sceneMat = [self cvMatFromUIImage:scene];
    
    // grey scale
    cvtColor(sceneMat, sceneMat, CV_BGR2GRAY);
    
    // filters for corners
    // https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    /*
    Mat imgSmoothed;
    normalize(sceneMat, sceneMat, 0, 255, NORM_MINMAX);
    GaussianBlur(sceneMat, imgSmoothed, cv::Size(0, 0), 3);
    addWeighted(sceneMat, 1.5, imgSmoothed, -0.5, 0, sceneMat);
    */
    // create ORB feature/descriptor detector
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(MAX_KP);
    
    std::vector<cv::KeyPoint> sceneKp;
    cv::Mat sceneDsc;
    
    orb->detect(sceneMat, sceneKp);
    
    adaptiveNonMaximalSuppresion(sceneKp, MAX_KP_FILTER_CONFIRM);
    
    orb->compute(sceneMat, sceneKp, sceneDsc);
    
    if (!sceneDsc.isContinuous()) {
        return nil;
    }
    
    // find matches
    FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
    
    // Detect
    
    int n = 0;
    
    for (int i = 0; i<[self targetDescriptors]->size(); i++) {
        int targetId = [self targetIds]->at(i);
        if (targetId == objectId) {
            n = i;
        }
    }
    
    std::vector<std::vector<DMatch>> matches;
    
    std::vector<cv::KeyPoint> kp = [self targetKeypoints]->at(n);
    cv::Mat dsc = [self targetDescriptors]->at(n);
    int targetId = [self targetIds]->at(n);
    
    // find matches
    matcher.knnMatch(dsc, sceneDsc, matches, 2);
    
    // filter for best matches
    std::vector<DMatch> good_matches;
    
    // D.Lowe ratio test for best results
    for (int i = 0; i < matches.size(); i++)
    {
        const float ratio = 0.7; // As in Lowe's paper; can be tuned
        // sometimes it is 0
        if (matches[i].size() == 2) {
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
    }
    
    int matchesCount = int(good_matches.size());
    
    // we got a match (probably)
    if (matchesCount >= MIN_MATCHES_TO_DETECT) {
        
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;
        
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back( kp[ good_matches[i].queryIdx ].pt );
            scene.push_back( sceneKp[ good_matches[i].trainIdx ].pt );
        }
        
        // find homography
        Mat H = findHomography( Mat(obj), Mat(scene), CV_RANSAC );
        
        // it may be empty (not found)
        if (H.cols > 0) {
            //-- Get the corners from the image_1 ( the object to be "detected" )
            std::vector<Point2f> obj_corners(4);
            // target size is always 400x400
            obj_corners[0] = cvPoint(0, 0);
            obj_corners[1] = cvPoint(IMAGE_SIZE, 0);
            obj_corners[2] = cvPoint(IMAGE_SIZE, IMAGE_SIZE);
            obj_corners[3] = cvPoint(0, IMAGE_SIZE);
            std::vector<Point2f> scene_corners(4);
            
            perspectiveTransform( Mat(obj_corners), Mat(scene_corners), H);
            
            // find bounds (we don't care about rotation)
            float x = 1000;
            float y = 1000;
            float w = 0;
            float h = 0;
            
            for (int i=0; i<4; i++) {
                x = min(x, scene_corners[i].x);
                y = min(y, scene_corners[i].y);
                w = max(w, scene_corners[i].x);
                h = max(h, scene_corners[i].y);
            }
            
            CGPoint center = CGPointMake((x+w)/2, (y+h)/2);
            
            // make it a little bit smaller
            float radius = 0.7 * (w-x)/2;
            
            CGRect bounds = CGRectMake(center.x - radius, center.y - radius, radius*2, radius*2);
            
            DetectResults *res = [[DetectResults alloc] initWithMatches:matchesCount forObject:targetId withBounds:bounds];
            return res;
        }
    }

    return nil;
    
}





/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
// PRIVATES
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

// C++
// http://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/#93395
//

void findMatches(DescriptorMatcher* matcher, cv::Mat targetDsc, cv::Mat sceneDsc, std::vector<DMatch>& good_matches)
{
    std::vector<std::vector<DMatch>> matches;
    
    // find matches
    matcher->knnMatch(targetDsc, sceneDsc, matches, 2);
    
    // D.Lowe ratio test for best results
    for (int i = 0; i < matches.size(); i++)
    {
        const float ratio = 0.7; // As in Lowe's paper; can be tuned
        // sometimes it is 0
        if (matches[i].size() == 2) {
            if (matches[i][0].distance < ratio * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
    }
}



void findFeatures(Mat objectMat, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int maxKps = 2000, int maxFinalKps = 100)
{
    Mat img_object;
    
    // grey scale
    cv::cvtColor(objectMat, img_object, CV_BGR2GRAY);
    
    // filters for corners
    // https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    
    Mat imgSmoothed;
    normalize(img_object, img_object, 0, 255, NORM_MINMAX);
    GaussianBlur(img_object, imgSmoothed, cv::Size(0, 0), 3);
    addWeighted(img_object, 1.5, imgSmoothed, -0.5, 0, img_object);
    
    // create ORB feature/descriptor detector
    Ptr<ORB> orb = ORB::create();
    orb->setMaxFeatures(maxKps);
    
    orb->detect(img_object, keypoints);
    
    adaptiveNonMaximalSuppresion(keypoints, maxFinalKps);
    
    orb->compute(img_object, keypoints, descriptors);
}


void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints, const int numToKeep )
{
    if( keypoints.size() < numToKeep ) { return; }
    
    //
    // Sort by response
    //
    std::sort( keypoints.begin(), keypoints.end(),
              [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
              {
                  return lhs.response > rhs.response;
              } );
    
    std::vector<cv::KeyPoint> anmsPts;
    
    std::vector<double> radii;
    radii.resize( keypoints.size() );
    std::vector<double> radiiSorted;
    radiiSorted.resize( keypoints.size() );
    
    const float robustCoeff = 1.11; // see paper
    
    for( int i = 0; i < keypoints.size(); ++i )
    {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( int j = 0; j < i && keypoints[j].response > response; ++j )
        {
            radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
    }
    
    std::sort( radiiSorted.begin(), radiiSorted.end(),
              [&]( const double& lhs, const double& rhs )
              {
                  return lhs > rhs;
              } );
    
    const double decisionRadius = radiiSorted[numToKeep];
    for( int i = 0; i < radii.size(); ++i )
    {
        if( radii[i] >= decisionRadius )
        {
            anmsPts.push_back( keypoints[i] );
        }
    }
    
    anmsPts.swap( keypoints );
}


//

- (CGRect) findHomography: (std::vector<DMatch>)matches withTarget: (std::vector<cv::KeyPoint>)targetKps onScene: (std::vector<cv::KeyPoint>)sceneKps {
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    for( int i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( targetKps[ matches[i].queryIdx ].pt );
        scene.push_back( sceneKps[ matches[i].trainIdx ].pt );
    }
    
    // find homography
    Mat H = findHomography( Mat(obj), Mat(scene), CV_RANSAC );
    
    // it may be empty (not found)
    if (H.cols > 0) {
        //-- Get the corners from the image_1 ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        // target size is always 400x400
        obj_corners[0] = cvPoint(0, 0);
        obj_corners[1] = cvPoint(IMAGE_SIZE, 0);
        obj_corners[2] = cvPoint(IMAGE_SIZE, IMAGE_SIZE);
        obj_corners[3] = cvPoint(0, IMAGE_SIZE);
        std::vector<Point2f> scene_corners(4);
        
        perspectiveTransform( Mat(obj_corners), Mat(scene_corners), H);
        
        // find bounds (we don't care about rotation)
        float x = 1000;
        float y = 1000;
        float w = 0;
        float h = 0;
        
        for (int i=0; i<4; i++) {
            x = min(x, scene_corners[i].x);
            y = min(y, scene_corners[i].y);
            w = max(w, scene_corners[i].x);
            h = max(h, scene_corners[i].y);
        }
        
        CGPoint center = CGPointMake((x+w)/2, (y+h)/2);
        
        // make it a little bit smaller
        float radius = 0.7 * (w-x)/2;
        
        CGRect bounds = CGRectMake(center.x - radius, center.y - radius, radius*2, radius*2);
        return bounds;
    }
    
    return CGRectZero;
}



//
// !! In OpenCV all operations on Images are done on Mat (Matrixes) !!
//

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

//

-(UIImage *)imageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                              //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}



@end
