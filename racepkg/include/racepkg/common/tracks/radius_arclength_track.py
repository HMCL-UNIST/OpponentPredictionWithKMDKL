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
from numpy import linalg as la
import casadi as ca
from racepkg.common.pytypes import VehicleState
import copy


class RadiusArclengthTrack():
    def __init__(self, track_width=None, slack=None, cl_segs=None):
        self.track_width = track_width
        self.slack = slack
        self.cl_segs = cl_segs
        self.n_segs = None
        self.key_pts = None
        self.track_length = None
        self.track_extents = None
        self.phase_out = False
        self.circuit = True  # Flag for whether the track is a circuit

    def initialize(self,  track_width=None, slack=None, cl_segs=None, init_pos=(0, 0, 0)):
        if track_width is not None:
            self.track_width = track_width
        if slack is not None:
            self.slack = slack
        if cl_segs is not None:
            self.cl_segs = cl_segs

        self.half_width = self.track_width / 2
        self.n_segs = self.cl_segs.shape[0]
        self.key_pts = self.get_track_key_pts(self.cl_segs, init_pos)

         
        self.track_length = self.key_pts[-1, 3]

        # Get the x-y extents of the track
        s_grid = np.linspace(0, self.track_length, int(10 * self.track_length))
        x_grid, y_grid = [], []
        for s in s_grid:
            xp, yp, _ = self.local_to_global((s, self.half_width + self.slack, 0))
            xm, ym, _ = self.local_to_global((s, -self.half_width - self.slack, 0))
            x_grid.append(xp)
            x_grid.append(xm)
            y_grid.append(yp)
            y_grid.append(ym)
        self.track_extents = dict(x_min=np.amin(x_grid), x_max=np.amax(x_grid), y_min=np.amin(y_grid),
                                  y_max=np.amax(y_grid))        
        ######################## track is doubled ##########################
        return

    def global_to_local_typed(self, data):  # data is vehicleState
        xy_coord = (data.x.x, data.x.y, data.e.psi)
        cl_coord = self.global_to_local(xy_coord)
        if cl_coord:
            data.p.s = cl_coord[0]
            data.p.x_tran = cl_coord[1]
            data.p.e_psi = cl_coord[2]
            return 0
        return -1

    def local_to_global_typed(self, data):
        cl_coord = (data.p.s, data.p.x_tran, data.p.e_psi)
        xy_coord = self.local_to_global(cl_coord)
        if xy_coord:
            data.x.x = xy_coord[0]
            data.x.y = xy_coord[1]
            data.e.psi = xy_coord[2]
            return -1
        return 0

    def get_curvature(self, s):
        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length
        key_pt_idx_s = np.where(s >= self.key_pts[:, 3])[0][-1]
        # d = s - self.key_pts[key_pt_idx_s, 3] # Distance along current segment
        return self.key_pts[key_pt_idx_s + 1, 5]  # curvature at this keypoint

    def update_curvature(self, state: VehicleState):
        for i in range(int(state.lookahead.n)):
            state.lookahead.curvature[i] = self.get_curvature(state.p.s + state.lookahead.dl * i)

    def get_curvature_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length) + self.track_length, self.track_length)
        # Piecewise constant function mapping s to track curvature
        pw_const_curvature = ca.pw_const(sym_s_bar, self.key_pts[1:-1, 3], self.key_pts[1:, 5])
        return ca.Function('track_curvature', [sym_s], [pw_const_curvature])

    def get_curvature_casadi_fn_dynamic(self):
        sym_s = ca.SX.sym('s', 1)
        track_length = ca.SX.sym('track_length', 1)
        key_pts = ca.SX.sym('key_pts', 5, 6)
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, track_length) + track_length, track_length)
        # Piecewise constant function mapping s to track curvature
        pw_const_curvature = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 5])
        return ca.Function('track_curvature_dyn', [sym_s, track_length, key_pts], [pw_const_curvature])

    def get_centerline_xy_from_s_casadi(self, s, track_length, key_pts, all_tracks):
        if not all_tracks:
            track_length = self.track_length
            key_pts = self.key_pts
        # TODO Replace with true if and only do the computations necessary
        def wrap_angle_ca(theta):
            return ca.if_else(theta < -ca.pi, 2*ca.pi + theta, ca.if_else(theta > ca.pi, theta - 2 * ca.pi, theta))
        sym_s_bar = ca.mod(ca.mod(s, track_length) + track_length, track_length)
        x_s = ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 0])
        y_s = ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 1])
        psi_s = ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 2])

        x_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 0])
        y_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 1])
        psi_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 2])
        curve_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 5])

        l = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 4])
        d = sym_s_bar - ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 3])

        # FIXME this is just to make sure no inf/nan occurs
        l = ca.if_else(l == 0, 1, l)

        r = ca.if_else(curve_f==0, 1, 1/curve_f)
        sgn = ca.sign(r)
        x_c = x_s + ca.fabs(r)*ca.cos(psi_s + sgn*ca.pi/2)
        y_c = y_s + ca.fabs(r)*ca.sin(psi_s + sgn*ca.pi/2)
        span_ang = d/ca.fabs(r)
        ang_norm = wrap_angle_ca(psi_s + sgn * ca.pi / 2)
        ang = -ca.sign(ang_norm) * (ca.pi - ca.fabs(ang_norm))
        psi_ = wrap_angle_ca(psi_s + sgn * span_ang)

        psi__ = wrap_angle_ca(psi_f)
        x_ = x_c + ca.fabs(r) * ca.cos(ang + sgn*span_ang)
        y_ = y_c + ca.fabs(r) * ca.sin(ang + sgn * span_ang)
        x__ = x_s + (x_f-x_s)*d/l
        y__ = y_s + (y_f-y_s)*d/l
        x = ca.if_else(curve_f == 0, x__, x_)
        y = ca.if_else(curve_f == 0, y__, y_)
        psi = ca.if_else(curve_f == 0, psi__, psi_)
        return (x, y, psi)

    def get_tangent_angle_casadi_fn(self):
        sym_s = ca.SX.sym('s', 1)
        # abs_angs = copy.copy(self.key_pts[:,2])
        seg_len = copy.copy(self.key_pts[:, 4])
        curvature = copy.copy(self.key_pts[:, 5])

        abs_angs = np.zeros(self.key_pts.shape[0] + 1)
        for i in range(self.key_pts.shape[0]):
            if curvature[i] == 0:
                abs_angs[i + 1] = abs_angs[i]
            else:
                abs_angs[i + 1] = abs_angs[i] + seg_len[i] * curvature[i]
        abs_angs = abs_angs[1:]
        # if self.circuit:
        #     abs_angs[-2:] = abs_angs[0] + 2*np.pi # Assumes that the last track segment is straight
        # Makes sure s is within [0, track_length]
        sym_s_bar = ca.mod(ca.mod(sym_s, self.track_length) + self.track_length, self.track_length)
        # Piecewise linear function mapping s to track tangent angle
        pw_lin_tangent_ang = ca.pw_lin(sym_s_bar, self.key_pts[:, 3], abs_angs)
        return ca.Function('track_tangent', [sym_s], [pw_lin_tangent_ang])

    def get_halfwidth(self, s):
        return self.half_width

    def get_track_key_pts(self, cl_segs, init_pos):
        if cl_segs is None:
            raise ValueError('Track segments have not been defined')

        n_segs = cl_segs.shape[0]
        # Given the segments in cl_segs we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
        # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
        # we compute also the cumulative length at the starting point of the segment at signed curvature
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        track_key_pts = np.zeros((n_segs + 1, 6))
        track_key_pts[0, 0] = init_pos[0]
        track_key_pts[0, 1] = init_pos[1]
        track_key_pts[0, 2] = init_pos[2]
        for i in range(1, n_segs + 1):
            x_prev = track_key_pts[i - 1, 0]
            y_prev = track_key_pts[i - 1, 1]
            psi_prev = track_key_pts[i - 1, 2]
            cum_s_prev = track_key_pts[i - 1, 3]

            l = cl_segs[i - 1, 0]
            r = cl_segs[i - 1, 1]

            if r == 0:
                # No curvature (i.e. straight line)
                psi = psi_prev
                x = x_prev + l * np.cos(psi_prev)
                y = y_prev + l * np.sin(psi_prev)
                curvature = 0
            else:
                # dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_prev - r * (np.sin(psi_prev))
                y_c = y_prev + r * (np.cos(psi_prev))
                # Angle spanned by curved segment
                theta = l / r
                # end of curve
                x = x_c + r * np.sin(psi_prev + theta)
                y = y_c - r * np.cos(psi_prev + theta)
                # curvature of segment
                curvature = 1 / r

                # next angle
                psi = wrap_angle(psi_prev + theta)
            cum_s = cum_s_prev + l
            track_key_pts[i] = np.array([x, y, psi, cum_s, l, curvature])

        return track_key_pts


    def remove_phase_out(self):
        if self.phase_out:
            self.track_length = self.key_pts[-2][3]
            self.key_pts = self.key_pts[0:-1]
            self.n_segs = self.n_segs - 1
            self.cl_segs = self.cl_segs[0:-1]
            self.phase_out = False

    def global_to_local_2(self, xy_coord):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        x, y, psi = xy_coord

        pos_cur = np.array([x, y])

        d2seg = np.zeros(self.key_pts.shape[0] - 1)
        for i in range(1, self.key_pts.shape[0]):
            x_s = self.key_pts[i - 1, 0]
            y_s = self.key_pts[i - 1, 1]
            psi_s = self.key_pts[i - 1, 2]
            x_f = self.key_pts[i, 0]
            y_f = self.key_pts[i, 1]
            curve_f = self.key_pts[i, 5]

            l = self.key_pts[i, 4]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            e_y = np.inf
            if curve_f == 0:
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2 and np.abs(
                        compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2:
                    # Check if on straight segment
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v) * np.sin(ang)
            else:
                # Check if on curved segment
                r = 1 / curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
                y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
                curve_center = np.array([x_c, y_c])

                span_ang = l / r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir) * (la.norm(v) - np.abs(r))
            d2seg[i - 1] = e_y

        seg_idx = np.argmin(np.abs(d2seg)) + 1

        x_s = self.key_pts[seg_idx - 1, 0]
        y_s = self.key_pts[seg_idx - 1, 1]
        psi_s = self.key_pts[seg_idx - 1, 2]
        s_s = self.key_pts[seg_idx - 1, 3]
        x_f = self.key_pts[seg_idx, 0]
        y_f = self.key_pts[seg_idx, 1]
        curve_f = self.key_pts[seg_idx, 5]

        pos_s = np.array([x_s, y_s])
        pos_f = np.array([x_f, y_f])

        if curve_f == 0:
            # Check if on straight segment
            v = pos_cur - pos_s
            ang = compute_angle(pos_s, pos_f, pos_cur)
            d = la.norm(v) * np.cos(ang)
        else:
            # Check if on curved segment
            r = 1 / curve_f
            dir = np.sign(r)

            # Find coordinates for center of curved segment
            x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
            y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
            curve_center = np.array([x_c, y_c])
            cur_ang = compute_angle(curve_center, pos_s, pos_cur)
            d = np.abs(cur_ang) * np.abs(r)

        return (s_s + d, d2seg[seg_idx - 1], 0)

    """
    Coordinate transformation from inertial reference frame (x, y, psi) to curvilinear reference frame (s, e_y, e_psi)
    Input:
        (x, y, psi): position in the inertial reference frame
    Output:
        (s, e_y, e_psi): position in the curvilinear reference frame
    """

    def global_to_local(self, xy_coord, line='center'):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        x = xy_coord[0]
        y = xy_coord[1]
        psi = xy_coord[2]

        pos_cur = np.array([x, y])
        cl_coord = None

        for i in range(1, self.key_pts.shape[0]):
            # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
            x_s = self.key_pts[i - 1, 0]
            y_s = self.key_pts[i - 1, 1]
            psi_s = self.key_pts[i - 1, 2]
            curve_s = self.key_pts[i - 1, 5]
            x_f = self.key_pts[i, 0]
            y_f = self.key_pts[i, 1]
            psi_f = self.key_pts[i, 2]
            curve_f = self.key_pts[i, 5]

            l = self.key_pts[i, 4]

            pos_s = np.array([x_s, y_s])
            pos_f = np.array([x_f, y_f])

            # Check if at any of the segment start or end points
            if la.norm(pos_s - pos_cur) == 0:
                # At start of segment
                s = self.key_pts[i - 1, 3]
                e_y = 0
                e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                cl_coord = (s, e_y, e_psi)                
                break
            if la.norm(pos_f - pos_cur) == 0:
                # At end of segment
                s = self.key_pts[i, 3]
                e_y = 0
                e_psi = np.unwrap([psi_f, psi])[1] - psi_f
                cl_coord = (s, e_y, e_psi)                
                break

            if curve_f == 0:                
                # Check if on straight segment
                if np.abs(compute_angle(pos_s, pos_cur, pos_f)) <= np.pi / 2 and np.abs(
                        compute_angle(pos_f, pos_cur, pos_s)) <= np.pi / 2:
                    v = pos_cur - pos_s
                    ang = compute_angle(pos_s, pos_f, pos_cur)
                    e_y = la.norm(v) * np.sin(ang)
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                  
                    
                    if np.abs(e_y) <= self.track_width / 2 + self.slack:
                        d = la.norm(v) * np.cos(ang)
                        s = self.key_pts[i - 1, 3] + d
                        e_psi = np.unwrap([psi_s, psi])[1] - psi_s
                        cl_coord = (s, e_y, e_psi)                        
                        break
                    else:                        
                        continue
                else:
                    continue                
            else:
                # Check if on curved segment                                
                r = 1 / curve_f
                dir = np.sign(r)

                # Find coordinates for center of curved segment
                x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
                y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)
                curve_center = np.array([x_c, y_c])

                span_ang = l / r
                cur_ang = compute_angle(curve_center, pos_s, pos_cur)
                if np.sign(span_ang) == np.sign(cur_ang) and np.abs(span_ang) >= np.abs(cur_ang):
                    v = pos_cur - curve_center
                    e_y = -np.sign(dir) * (la.norm(v) - np.abs(r))
                    # Check if deviation from centerline is within track width plus some slack for current segment
                    # (allows for points outside of track boundaries)
                    if np.abs(e_y) <= self.track_width / 2 + self.slack:
                        d = np.abs(cur_ang) * np.abs(r)
                        s = self.key_pts[i - 1, 3] + d
                        e_psi = np.unwrap([psi_s + cur_ang, psi])[1] - (psi_s + cur_ang)
                        cl_coord = (s, e_y, e_psi)                        
                        break
                    else:                        
                        continue
                else:                    
                    continue

        if line == 'inside':
            cl_coord = (cl_coord[0], cl_coord[1] - self.track_width / 3, cl_coord[2])
            
            
        elif line == 'outside':
            cl_coord = (cl_coord[0], cl_coord[1] + self.track_width / 3, cl_coord[2])
        elif line == 'pid_offset':
            # PID controller tends to cut to the inside of the track
            cl_coord = (cl_coord[0], cl_coord[1] + (0.1 * self.track_width / 2), cl_coord[2])
        
        # if cl_coord is None:
        #     raise ValueError('Point is out of the track!')
        return cl_coord

    """
    Coordinate transformation from curvilinear reference frame (s, e_y, e_psi) to inertial reference frame (x, y, psi)
    Input:
        (s, e_y, e_psi): position in the curvilinear reference frame
    Output:
        (x, y, psi): position in the inertial reference frame
    """
    def local_to_global_ca(self, s, x_tran, e_psi, key_pts, track_length, all_tracks=False):
        if not all_tracks:
            track_length = self.track_length
            key_pts = self.key_pts
            # TODO Replace with true if and only do the computations necessary

        def wrap_angle_ca(theta):
            return ca.if_else(theta < -ca.pi, 2 * ca.pi + theta, ca.if_else(theta > ca.pi, theta - 2 * ca.pi, theta))

        sym_s_bar = ca.mod(ca.mod(s, track_length) + track_length, track_length)
        x_s = ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 0])
        y_s = ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 1])
        psi_s = ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 2])

        x_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 0])
        y_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 1])
        psi_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 2])
        curve_f = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 5])

        l = ca.pw_const(sym_s_bar, key_pts[1:-1, 3], key_pts[1:, 4])
        d = sym_s_bar - ca.pw_const(sym_s_bar, key_pts[1:, 3], key_pts[:, 3])

        # FIXME this is just to make sure no inf/nan occurs
        l = ca.if_else(l == 0, 1, l)

        r = ca.if_else(curve_f == 0, 1, 1 / curve_f)
        sgn = ca.sign(r)

        x_c = x_s + ca.fabs(r) * ca.cos(psi_s + sgn * ca.pi / 2)
        y_c = y_s + ca.fabs(r) * ca.sin(psi_s + sgn * ca.pi / 2)
        span_ang = d / ca.fabs(r)
        ang_norm = wrap_angle_ca(psi_s + sgn * ca.pi / 2)
        ang = -ca.sign(ang_norm) * (ca.pi - ca.fabs(ang_norm))
        psi_ = wrap_angle_ca(psi_s + sgn * span_ang + e_psi)

        psi__ = wrap_angle_ca(psi_f + e_psi)
        x_ = x_c + (ca.fabs(r) - sgn*x_tran)* ca.cos(ang + sgn * span_ang)
        y_ = y_c + (ca.fabs(r) - sgn*x_tran) * ca.sin(ang + sgn * span_ang)
        x__ = x_s + (x_f - x_s) * d / l + x_tran*ca.cos(psi_f+ ca.pi/2)
        y__ = y_s + (y_f - y_s) * d / l + x_tran*ca.sin(psi_f+ ca.pi/2)
        x = ca.if_else(curve_f == 0, x__, x_)
        y = ca.if_else(curve_f == 0, y__, y_)
        psi = ca.if_else(curve_f == 0, psi__, psi_)
        return (x, y, psi)

    def local_to_global(self, cl_coord):
        if self.key_pts is None:
            raise ValueError('Track key points have not been defined')

        # s = np.mod(cl_coord[0], self.track_length) # Distance along current lap
        s = cl_coord[0]
        while s < 0: s += self.track_length
        while s >= self.track_length: s -= self.track_length

        e_y = cl_coord[1]
        e_psi = cl_coord[2]

        # Find key point indicies corresponding to current segment
        # key_pts = [x, y, psi, cumulative length, segment length, signed curvature]
        key_pt_idx_s = np.where(s >= self.key_pts[:, 3])[0][-1]
        key_pt_idx_f = key_pt_idx_s + 1
        seg_idx = key_pt_idx_s

        x_s = self.key_pts[key_pt_idx_s, 0]
        y_s = self.key_pts[key_pt_idx_s, 1]
        psi_s = self.key_pts[key_pt_idx_s, 2]
        curve_s = self.key_pts[key_pt_idx_s, 5]
        x_f = self.key_pts[key_pt_idx_f, 0]
        y_f = self.key_pts[key_pt_idx_f, 1]
        psi_f = self.key_pts[key_pt_idx_f, 2]
        curve_f = self.key_pts[key_pt_idx_f, 5]

        l = self.key_pts[key_pt_idx_f, 4]
        d = s - self.key_pts[key_pt_idx_s, 3]  # Distance along current segment

        if curve_f == 0:
            # Segment is a straight line
            x = x_s + (x_f - x_s) * d / l + e_y * np.cos(psi_f + np.pi / 2)
            y = y_s + (y_f - y_s) * d / l + e_y * np.sin(psi_f + np.pi / 2)
            psi = wrap_angle(psi_f + e_psi)
        else:
            r = 1 / curve_f
            dir = sign(r)

            # Find coordinates for center of curved segment
            x_c = x_s + np.abs(r) * np.cos(psi_s + dir * np.pi / 2)
            y_c = y_s + np.abs(r) * np.sin(psi_s + dir * np.pi / 2)

            # Angle spanned up to current location along segment
            span_ang = d / np.abs(r)

            # Angle of the tangent vector at the current location
            psi_d = wrap_angle(psi_s + dir * span_ang)

            ang_norm = wrap_angle(psi_s + dir * np.pi / 2)
            ang = -sign(ang_norm) * (np.pi - np.abs(ang_norm))

            x = x_c + (np.abs(r) - dir * e_y) * np.cos(ang + dir * span_ang)
            y = y_c + (np.abs(r) - dir * e_y) * np.sin(ang + dir * span_ang)
            psi = wrap_angle(psi_d + e_psi)
        return (x, y, psi)


def wrap_angle(theta):
    if theta < -np.pi:
        wrapped_angle = 2 * np.pi + theta
    elif theta > np.pi:
        wrapped_angle = theta - 2 * np.pi
    else:
        wrapped_angle = theta

    return wrapped_angle


def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res


"""
Helper function for computing the angle between the vectors point_1-point_0
and point_2-point_0. All points are defined in the inertial frame
Input:
    point_0: position of the intersection point (np.array of size 2)
    point_1, point_2: defines the intersecting lines (np.array of size 2)
Output:
    theta: angle in radians
"""


def compute_angle(point_0, point_1, point_2):
    v_1 = point_1 - point_0
    v_2 = point_2 - point_0

    dot = v_1.dot(v_2)
    det = v_1[0] * v_2[1] - v_1[1] * v_2[0]
    theta = np.arctan2(det, dot)

    return theta
