#!/usr/bin/env python3
'''
MIT License

Copyright (c) 2022 Model Predictive Control (MPC) Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
from racepkg.common.tracks.radius_arclength_track import RadiusArclengthTrack

class StraightTrack(RadiusArclengthTrack):
    def __init__(self, length, width, slack, phase_out=False):
        if phase_out:
            cl_segs = np.array([[length, 0],
                                [10, 0]])
        else:
            cl_segs = np.array([length, 0]).reshape((1,-1))

        super().__init__(width, slack, cl_segs)
        self.phase_out = phase_out
        self.initialize()

class CurveTrack(RadiusArclengthTrack):
    def __init__(self, enter_straight_length, 
                        curve_length, 
                        curve_swept_angle, 
                        exit_straight_length, 
                        width, 
                        slack,
                        phase_out=False,
                        ccw=True):
        if ccw:
            s = 1
        else:
            s = -1
        if phase_out:
            cl_segs = np.array([[enter_straight_length, 0],
                            [curve_length,          s*curve_length/curve_swept_angle],
                            [exit_straight_length,  0],
                                [50, 0]])
        else:
            cl_segs = np.array([[enter_straight_length, 0],
                                [curve_length, s * curve_length / curve_swept_angle],
                                [exit_straight_length, 0]])
        super().__init__(width, slack, cl_segs)
        self.phase_out = phase_out
        self.initialize()

class ChicaneTrack(RadiusArclengthTrack):
    def __init__(self, enter_straight_length, 
                        curve1_length, 
                        curve1_swept_angle, 
                        mid_straight_length,
                        curve2_length,
                        curve2_swept_angle,
                        exit_straight_length,
                        width, 
                        slack,
                        phase_out=False,
                        mirror=False):
        if mirror:
            s1, s2 = 1, -1
        else:
            s1, s2 = -1, 1
        if phase_out:
            cl_segs = np.array([[enter_straight_length, 0],
                                [curve1_length, s1 * curve1_length / curve1_swept_angle],
                                [mid_straight_length, 0],
                                [curve2_length, s2 * curve2_length / curve2_swept_angle],
                                [exit_straight_length, 0],
                                [10, 0]])
        else:
            cl_segs = np.array([[enter_straight_length, 0],
                            [curve1_length,          s1*curve1_length/curve1_swept_angle],
                            [mid_straight_length,    0],
                            [curve2_length,          s2*curve2_length/curve2_swept_angle],
                            [exit_straight_length,   0]])

        super().__init__(width, slack, cl_segs)
        self.phase_out = phase_out
        self.initialize()