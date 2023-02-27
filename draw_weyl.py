# Packages:
# pip install drawSvg hyperbolic latextools fs numpy

import dataclasses
from dataclasses import dataclass, field
from typing import List, Set, Tuple

import math
import numpy as np
import fs.tempfs

import drawSvg as draw
from drawSvg.widgets import AsyncAnimation
from hyperbolic import euclid3d, euclid
import latextools


@dataclass
class WCConfig:
    scale: float = 500
    w: float = 0.6
    edge_stroke_width: float = 1.5
    edge_stroke_color: str = 'black'
    edge_stroke_dasharray: List = (6, 4)
    other_stroke_width: float = 1
    other_stroke_color: str = 'black'
    other_stroke_opacity: float = 0.5
    other_stroke_dasharray: List = (8, 6)
    arc_size: float = 0.015
    corner_size: float = 0.015
    arc_color: float = 'black'
    arc_opacity: float = 0.3
    label_size: float = 12
    label_color: str = 'black'
    traj_stroke_width: float = 3
    traj_stroke_color: str = 'lightblue'
    traj_mark_color: str = 'blue'
    default_view: 'ViewAngle' = field(default_factory=lambda: ViewAngle())
    default_multiview: 'ViewAngle' = field(default_factory=lambda: [
        ViewAngle(45+d, 10) for d in range(0, 360, 120)
    ])
    multi_spacing: float = -0.2
    crop: Tuple[float, float, float, float] = (0,)*4  # Top, bottom, left, right

    # Corners of the chamber
    corners: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0,0,0),
        (1/2,0,0),
        (1/4,1/4,0),
        (1/4,1/4,1/4),
    ])
    # Edges of the chamber
    edges: List[List[Tuple[float, float, float]]] = None
    # CCW faces 0-bottom, 1-back-right, 2-back-left, 3-top
    faces: List[List[Tuple[float, float, float]]] = None
    # Non-edge lines on the chamber
    lines: List[Tuple[Tuple, Tuple, int]] = field(default_factory=lambda: [
        [(1/4,0,0), (1/8,1/8,0), 0],  # p1, p2, face
        [(1/4,0,0), (3/8,1/8,0), 0],
        [(1/4,0,0), (1/8,1/8,1/8), 3],
        [(1/4,0,0), (3/8,1/8,1/8), 3],
        [(1/4,1/4,0), (1/8,1/8,1/8), 2],
        [(1/4,1/4,0), (3/8,1/8,1/8), 1],
        [(1/8,1/8,0), (1/8,1/8,1/8), 2],
        [(3/8,1/8,0), (3/8,1/8,1/8), 1],
        [(3/8,1/8,1/8), (1/8,1/8,1/8), 3],
    ])
    labels_plain: List[Tuple[Tuple, str, List[int]]] = field(
            default_factory=lambda: [
        [(0,0,0), 'I0', (0, 2, 3)],  # point, label, (face1, ...)
        [(1/2,0,0), 'I1', (0, 1, 3)],
        [(1/4,0,0), 'CZ/CNOT', (0, 3)],
        [(1/4,1/4,1/4), 'SWAP', (1, 2, 3)],
        [(1/4,1/4,0), 'iSWAP', (0, 1, 2)],
        [(1/8,1/8,0), '√iSWAP', (0, 2)],
        [(3/8,1/8,0), '√iSWAP', (0, 1)],
        [(1/8,1/8,1/8), '√SWAP', (2, 3)],
        [(3/8,1/8,1/8), '√SWAP^†', (1, 3)],
    ])
    labels_latex: List[Tuple[Tuple, str, List[int]]] = field(
            default_factory=lambda: [
        [(0,0,0), r'$\text{I}_0$', (0, 2, 3)],  # point, label, (face1, ...)
        [(1/2,0,0), r'$\text{I}_1$', (0, 1, 3)],
        [(1/4,0,0), r'$\text{CZ/CNOT}$', (0, 3)],
        [(1/4,1/4,1/4), r'$\text{SWAP}$', (1, 2, 3)],
        [(1/4,1/4,0), r'$\text{iSWAP}$', (0, 1, 2)],
        [(1/8,1/8,0), r'$\sqrt{\text{iSWAP}}$', (0, 2)],
        [(3/8,1/8,0), r'$\sqrt{\text{iSWAP}}$', (0, 1)],
        [(1/8,1/8,1/8), r'$\sqrt{\text{SWAP}}$', (2, 3)],
        [(3/8,1/8,1/8), r'$\sqrt{\text{SWAP}^\dagger}$', (1, 3)],
    ])
    corner_arcs: List[Tuple[Tuple, Tuple, Tuple, Tuple, int, int, int]] = field(
            default_factory=lambda: [
        # Args to draw_corner_arc
        [[0,0,0],      [1,0,0],   [1,1,0],  [1,1,1],  0, 2, 3],  # I0
        [[1/2,0,0],    [-1,0,0],  [-1,1,0], [-1,1,1], 0, 1, 3],  # I1
        [[1/4,1/4,1/4],[-1,-1,-1],[1,-1,-1],[0,0,-1], 3, 1, 2],  # SWAP
        [[1/4,1/4,0],  [-1,-1,0], [1,-1,0], [0,0,1],  0, 1, 2],  # iSWAP
    ])
    edge_arcs: List[Tuple[Tuple, Tuple, Tuple, Tuple, int, int]] = field(
            default_factory=lambda: [
        # Args to draw_edge_arc
        [[1/8,1/8,1/8],[-1,-1,-1],[1,1,-1], [1,-1,-1],  2, 3],  # sqrt-SWAP
        [[3/8,1/8,1/8],[1,-1,-1], [-1,1,-1],[-1,-1,-1], 1, 3],  # sqrt-SWAP^t
        [[1/4,0,0],    [1,0,0],   [0,1,0],  [0,1,1],    0, 3],  # CNOT
        [[1/8,1/8,0],  [1,1,0],   [0,0,1],  [1,-1,0],   2, 0],  # sqrt-iSWAP
        [[3/8,1/8,0],  [-1,1,0],  [0,0,1],  [-1,-1,0],  1, 0],  # sqrt-iSWAP
    ])
    def __post_init__(self):
        pts = self.corners
        self.edges = [[p1, p2] for i, p1 in enumerate(pts) for p2 in pts[i+1:]]
        self.faces = [(list(pts)*2)[i:i+3][::1 if i%2 else -1]  # CCW faces
                      for i in range(len(pts))]


@dataclass
class ViewAngle:
    rotation_deg: float = 30
    tilt_deg: float = 10
    fov_deg: float = 45

    def rotation(self):
        return (
            euclid3d.rotation3d([0,1,0], math.radians(self.tilt_deg))
            @ euclid3d.rotation3d([0,0,1], math.radians(self.rotation_deg))
        )
    def transform(self, scale, w):
        return (
            euclid3d.scaling([scale/w]*3)
            @ euclid3d.axis_swap([1,2,0])
            @ self.rotation()
            @ euclid3d.translation([-1/4,-1/8,-1/8])
        )
    def perspective(self, scale):
        return euclid3d.perspective3d(math.radians(self.fov_deg), scale)
    def projection(self, scale, w):
        return self.perspective(scale) @ self.transform(scale, w)


def unit_vec(v):
    return v / np.linalg.norm(v)


@dataclass
class DrawContext:
    config: WCConfig
    for_latex: bool
    proj: euclid3d.Projection
    d: draw.Drawing
    g_back_front: List
    g_inside: draw.Group
    front_faces: Set[int] = None

    def _front_faces(context):
        proj = context.proj
        return set(
            i
            for i, f_pts in enumerate(context.config.faces)
            if np.cross(np.array(proj.p2(*f_pts[1]))-proj.p2(*f_pts[0]),
                        np.array(proj.p2(*f_pts[2]))-proj.p2(*f_pts[0])
                       ) >= 0
        )

    def _calc_dasharray(context, pt1, pt2, dasharray, off=False):
        if off: return None
        length = np.linalg.norm(np.array(pt1) - pt2)
        proj_len = np.linalg.norm(
                np.array(context.proj.p2(*pt1)) - context.proj.p2(*pt2))
        stroke_len = proj_len / length / context.config.scale * context.config.w
        # TODO: Adjust length of individual segments for perspective
        return ' '.join(f'{dash*stroke_len}' for dash in dasharray)

    def draw_edge_arc(context, center, dx, dy1, dy2, face1, face2):
        conf = context.config
        r = conf.arc_size
        center, dx, dy1, dy2 = np.array([center, dx, dy1, dy2], dtype=float)
        dx /= np.linalg.norm(dx)
        dy1 -= dx*np.dot(dx, dy1)
        dy2 -= dx*np.dot(dx, dy2)
        dx *= r
        dy1 *= r / np.linalg.norm(dy1)
        dy2 *= r / np.linalg.norm(dy2)
        for dy, fid in zip([dy1, dy2], [face1, face2]):
            el = euclid.shapes.EllipseArc.fromBoundingQuad(
                *context.proj.p2(*center+dx+dy),
                *context.proj.p2(*center+dx-dy),
                *context.proj.p2(*center-dx-dy),
                *context.proj.p2(*center-dx+dy),
                *context.proj.p2(*center-dx),
                *context.proj.p2(*center+dx),
                *context.proj.p2(*center+dy),
            )
            is_front = fid in context.front_faces
            context.g_back_front[is_front].append(
                p := draw.Path(opacity=conf.arc_opacity, stroke='none',
                               fill=conf.arc_color),
                z=11-(not is_front)*4)
            if el is not None:
                el.drawToPath(p)

    def draw_corner_arc(context, center, dx1, dx2, dx3, face1,
                        face2, face3):
        conf = context.config
        r = conf.corner_size
        center, dx1, dx2, dx3 = np.array([center, dx1, dx2, dx3], dtype=float)
        for dxa, dxb, fid in zip([dx1, dx2, dx3], [dx2, dx3, dx1],
                                 [face1, face2, face3]):
            dxa, dxb = unit_vec(dxa), unit_vec(dxb)
            dy = unit_vec(dxb - dxa*np.dot(dxa, dxb))
            el = euclid.shapes.EllipseArc.fromBoundingQuad(
                *context.proj.p2(*center+r*(dxa+dy)),
                *context.proj.p2(*center+r*(dxa-dy)),
                *context.proj.p2(*center+r*(-dxa-dy)),
                *context.proj.p2(*center+r*(-dxa+dy)),
                *context.proj.p2(*center+r*(dxb)),
                *context.proj.p2(*center+r*(dxa)),
                *context.proj.p2(*center+r*(-dxa)),
                excludeMid=True,
            )
            is_front = fid in context.front_faces
            context.g_back_front[is_front].append(
                p := draw.Path(opacity=conf.arc_opacity, stroke='none',
                               fill=conf.arc_color),
                z=11-(not is_front)*4)
            if el is not None:
                p.M(*context.proj.p2(*center))
                el.drawToPath(p, includeL=True)
                p.Z()


class WeylChamber:
    def __init__(self, config: WCConfig=WCConfig(), traj_points=(),
                 draw_extra=lambda context:None):
        self.config = config
        self.traj_points = traj_points
        self.draw_extra = draw_extra

    def as_widget(self, degs_per_sec=30, fps=10, view=None):
        if view is None:
            view = self.config.default_view
        if view == 'multi':
            view = self.config.default_multiview
        if isinstance(view, ViewAngle):
            view = dataclasses.replace(view)
        else:
            view = [dataclasses.replace(v) for v in view]
        widget = AsyncAnimation(fps=10)
        @widget.set_draw_frame
        def draw_frame(secs=0):
            if isinstance(view, ViewAngle):
                v = dataclasses.replace(
                        view, rotation_deg=view.rotation_deg+secs*degs_per_sec)
                return self.as_drawing(v, for_latex=False)
            else:
                v = [
                    dataclasses.replace(
                        v, rotation_deg=v.rotation_deg+secs*degs_per_sec)
                    for v in view
                ]
                return self.as_side_by_side_drawing(v, for_latex=False)
        return widget

    def as_drawing(self, view: ViewAngle=None, for_latex=False):
        if view is None:
            view = self.config.default_view
        scale, crop = self.config.scale, self.config.crop
        d = draw.Drawing(
                scale-sum(crop[2:4])*scale,
                scale-sum(crop[0:2])*scale,
                origin=(-scale/2+crop[2]*scale, -scale/2+crop[1]*scale),
                displayInline=False)
        self.add_to_drawing(d, view, for_latex)
        return d

    def as_side_by_side_drawing(self, views=None, for_latex=False):
        if views is None:
            views = self.config.default_multiview
        scale, crop = self.config.scale, self.config.crop
        d = draw.Drawing(
                scale+scale*(1+self.config.multi_spacing)*(len(views)-1)
                    - sum(crop[2:4])*scale,
                scale-sum(crop[0:2]*scale),
                origin=(-scale/2+crop[2]*scale, -scale/2+crop[1]*scale),
                displayInline=False)
        for i, v in enumerate(views):
            g = draw.Group()
            d.append(draw.Use(
                g, x=i*self.config.scale*(1+self.config.multi_spacing), y=0))
            self.add_to_drawing(g, v, for_latex)
        return d

    def add_to_drawing(self, d, view: ViewAngle=ViewAngle(), for_latex=False):
        d.append(back := draw.Group())
        d.append(inside := draw.Group())
        d.append(front := draw.Group())
        context = DrawContext(
            config = self.config,
            for_latex=for_latex,
            proj = view.projection(self.config.scale, self.config.w),
            d = d,
            g_back_front = (back, front),
            g_inside = inside,
        )
        context.front_faces = context._front_faces()

        conf = context.config
        for i, pt in enumerate(conf.corners):  # Vertices
            is_front = any(f in context.front_faces
                           for f, f_pts in enumerate(conf.faces)
                           if pt in f_pts)
            context.g_back_front[is_front].append(
                    draw.Circle(*context.proj.p2(*pt), conf.edge_stroke_width/2,
                                stroke='none', fill=conf.edge_stroke_color),
                    z=10)
        for pt1, pt2 in conf.edges:  # Edges
            is_front = any(f in context.front_faces
                           for f, f_pts in enumerate(conf.faces)
                           if pt1 in f_pts and pt2 in f_pts)
            context.g_back_front[is_front].append(
                    draw.Line(*context.proj.p2(*pt1), *context.proj.p2(*pt2),
                              stroke=conf.edge_stroke_color,
                              stroke_width=conf.edge_stroke_width,
                              stroke_dasharray=context._calc_dasharray(
                                  pt1, pt2, conf.edge_stroke_dasharray,
                                  off=is_front)),
                    z=9)
        for pt1, pt2, face_id in conf.lines:  # Other lines
            is_front = face_id in context.front_faces
            context.g_back_front[is_front].append(
                    draw.Line(*context.proj.p2(*pt1), *context.proj.p2(*pt2),
                              opacity=conf.other_stroke_opacity,
                              stroke=conf.other_stroke_color,
                              stroke_width=conf.other_stroke_width,
                              stroke_dasharray=context._calc_dasharray(
                                  pt1, pt2, conf.other_stroke_dasharray,
                                  off=is_front)),
                    z=9)
        for args in conf.corner_arcs:
            context.draw_corner_arc(*args)
        for args in conf.edge_arcs:
            context.draw_edge_arc(*args)

        labels = conf.labels_latex if context.for_latex else conf.labels_plain
        for pt, text, face_ids in labels:
            is_front = any(f in context.front_faces for f in face_ids)
            proj_pt = context.proj.p2(*pt)
            if proj_pt[0] < 0:
                anchor = 'end'
                if proj_pt[1] < -0.1*conf.scale/conf.w:
                    off = [-5, -5-12+2]
                else:
                    off = [-5, 5]
            else:
                anchor = 'start'
                if proj_pt[1] < -0.1*conf.scale/conf.w:
                    off = [5, -5-12+2]
                else:
                    off = [5, 5]
            context.g_back_front[is_front].append(
                    draw.Text(text, conf.label_size, *(np.array(proj_pt)+off),
                              stroke='none', fill=conf.label_color,
                              text_anchor=anchor),
                         z=200)

        # Trajectory lines and points
        self.draw_trajectory(context)
        self.draw_extra(context)

    def draw_trajectory(self, context):
        if len(self.traj_points) <= 0:
            return
        config, proj = context.config, context.proj

        context.g_inside.append(group := draw.Group())
        group.append(traj_path := draw.Path(
                stroke_width=config.traj_stroke_width,
                stroke=config.traj_stroke_color, fill='none',
                stroke_linejoin='round'))
        pt = self.traj_points[0]
        traj_path.M(*proj.p2(*pt))
        group.append(draw.Circle(*proj.p2(*pt), config.traj_stroke_width/2,
                                 stroke='none', fill=config.traj_mark_color),
                     z=proj.p3(*pt)[2])
        for pt in self.traj_points[1:]:
            traj_path.L(*proj.p2(*pt))
            group.append(draw.Circle(*proj.p2(*pt), config.traj_stroke_width/2,
                            stroke='none', fill=config.traj_mark_color),
                         z=proj.p3(*pt)[2])
