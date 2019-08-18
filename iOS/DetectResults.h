//
//  DetectResult.h
//  VisionAppTest
//
//  Created by Ufos on 11.11.2017.
//  Copyright © 2017 Panowie Programisci. All rights reserved.
//

#import <UIKit/UIKit.h>


@interface DetectResults : NSObject


@property int numberOfMatches;
@property int objectId;
@property CGRect bounds;

- (id) initWithMatches: (int)numberOfMatches forObject: (int)objectId withBounds: (CGRect)bounds;

@end


