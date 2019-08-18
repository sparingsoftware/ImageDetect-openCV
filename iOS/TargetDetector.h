//
//  TargetDetector.h
//  P1V0
//
//  Created by Ufos on 29.11.2017.
//  Copyright © 2017 Panowie Programisci. All rights reserved.
//


#import <UIKit/UIKit.h>
#import "DetectResults.h"


//

@interface TargetDetector : NSObject

- (BOOL) addTarget: (UIImage*)image;
- (void) loadTargets: (NSString*)filename;
- (NSArray<DetectResults*>*) detectTargets: (UIImage*)scene;

- (DetectResults*) detectTargetWithId: (int)objectId onScene: (UIImage*)scene;

- (NSArray<DetectResults*>*) detectTargetWithIds: (NSArray*)objecstId onScene: (UIImage*)scene;

@end

