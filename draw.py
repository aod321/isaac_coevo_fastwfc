import math
import numpy as np
from isaacgym import gymapi
from isaacgym.gymutil import LineGeometry, AxesGeometry, WireframeSphereGeometry, WireframeBoxGeometry, draw_lines


force_vector_color = gymapi.Vec3(0.7, 0.2, 0.15)

class FrustumGeometry(LineGeometry):

    def __init__(self, scale=1., aspect_ratio=None, pose=None, color=None):
        if color is None:
            color = (1, 0, 0)
        if aspect_ratio is None:
            aspect_ratio = 1.

        num_lines = 8

        x = 2*0.5 * scale * aspect_ratio
        y = 2*0.5 * scale
        z = scale

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        # projection frustum
        verts[0][0] = (0, 0, 0)
        verts[0][1] = (x, y, z)
        verts[1][0] = (0, 0, 0)
        verts[1][1] = (-x, y, z)
        verts[2][0] = (0, 0, 0)
        verts[2][1] = (x, -y, z)
        verts[3][0] = (0, 0, 0)
        verts[3][1] = (-x, -y, z)

        # imaging plane
        verts[4][0] = (-x, y, z)
        verts[4][1] = (x, y, z)
        verts[5][0] = (x, -y, z)
        verts[5][1] = (x, y, z)
        verts[6][0] = (-x, -y, z)
        verts[6][1] = (x, -y, z)
        verts[7][0] = (-x, -y, z)
        verts[7][1] = (-x, y, z)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


def draw_camera(
        gym_,
        viewer_,
        env_ptr,
        transform_,
        length=0.5,
        color=(1, 0, 0),
        frustum_aspect_ratio=None,
        draw_frustum=True,
        draw_housing=True
    ):

    transform = transform_

    transform.r *= gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 0.5*math.pi)

    if draw_housing:
        # scale housing based on length
        cam_xdim, cam_ydim, cam_zdim = (length, length, 2. * length)
        box_geom = WireframeBoxGeometry(
            xdim=cam_xdim,
            ydim=cam_ydim,
            zdim=cam_zdim,
            color=color
        )
    if draw_frustum:
        frust_geom = FrustumGeometry(length, frustum_aspect_ratio, color=color)

    # Draw camera "box" housing
    if draw_housing:
        # This pushes the housing backwards, so the triad is drawn at the housing end, not the middle
        cam_box_transform = transform * gymapi.Transform(p=gymapi.Vec3(0., 0., -cam_zdim / 2.))
        draw_lines(box_geom, gym_, viewer_, env_ptr, cam_box_transform)

    # Draw camera frustum
    if draw_frustum:
        draw_lines(frust_geom, gym_, viewer_, env_ptr, transform)
