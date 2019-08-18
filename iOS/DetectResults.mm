//
//  DetectResult.m
//  VisionAppTest
//
//  Created by Ufos on 11.11.2017.
//  Copyright © 2017 Panowie Programisci. All rights reserved.
//


#import "DetectResults.h"

@implementation DetectResults

- (id) initWithMatches: (int)numberOfMatches forObject: (int)objectId withBounds: (CGRect)bounds {
    
    self = [super init];
    
    if (self) {
        self.numberOfMatches = numberOfMatches;
        self.objectId = objectId;
        self.bounds = bounds;
    }
    
    return self;
}

@end
