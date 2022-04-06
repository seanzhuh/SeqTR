import cv2
import math
import numpy
from ..builder import PIPELINES


@PIPELINES.register_module()
class SampleMaskVertices(object):
    def __init__(self,
                 center_sampling=False,
                 num_ray=18):
        super().__init__()
        self.center_sampling = center_sampling
        assert num_ray > 0
        self.num_ray = num_ray

    # after padding
    def __call__(self, results):
        assert results['with_mask']
        gt_mask = results['gt_mask'].masks[0]
        center, contour, KEEP = self.get_mass_center(gt_mask)
        vertices = self.sample_mask_vertices(
            center, contour, KEEP, results['pad_shape'][:2])
        results['gt_mask_vertices'] = vertices
        results['mass_center'] = center
        return results

    def get_mass_center(self, mask):
        contour, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = sorted(
            contour, key=lambda x: cv2.contourArea(x), reverse=True)
        contour = contour[0][:, 0, :]  # x, y coordinate of contour
        contour_info = cv2.moments(contour)
        KEEP = False
        if contour_info['m00'] > 0.:
            KEEP = True
        if KEEP:
            mass_x = contour_info['m10'] / contour_info['m00']
            mass_y = contour_info['m01'] / contour_info['m00']
            center = numpy.array([mass_x, mass_y])
        else:
            center = numpy.array([-1., -1.])
        return center, contour, KEEP

    def sample_mask_vertices(self, center, contour, KEEP=True, max_shape=None):
        vertices = numpy.empty(
            (2, self.num_ray), dtype=numpy.float32)
        vertices.fill(-1)
        if not KEEP:
            return vertices
        num_pts = contour.shape[0]
        if num_pts <= self.num_ray:
            vertices[:, :num_pts] = contour.transpose()
            return vertices
        inside_contour = cv2.pointPolygonTest(contour, center, False) > 0
        if self.center_sampling and inside_contour:
            c_x, c_y = center
            x = contour[:, 0] - center[0]
            y = contour[:, 1] - center[1]
            angle = numpy.arctan2(y, x) * 180 / numpy.pi
            angle[angle < 0] += 360
            angle = angle.astype(numpy.uint32)
            distance = numpy.sqrt(x ** 2 + y ** 2)
            angles, distances = [], []
            for ang in range(0, 360, 360 // self.num_ray):
                if ang in angle:
                    dist = distance[ang == angle].max()
                    angles.append(ang)
                    distances.append(dist)
                else:
                    for increment in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
                        aux_ang = ang + increment
                        if aux_ang in angle:
                            dist = distance[aux_ang == angle].max()
                            angles.append(aux_ang)
                            distances.append(dist)
                            break
            angles = numpy.array(angles)
            distances = numpy.array(distances)
            angles = angles / 180 * numpy.pi
            sin = numpy.sin(angles)
            cos = numpy.cos(angles)
            vertex_x = c_x + distances * cos
            vertex_y = c_y + distances * sin
        else:
            interval = math.ceil(num_pts / self.num_ray)
            vertex_x = contour[::interval, 0]
            vertex_y = contour[::interval, 1]
        if max_shape is not None:
            vertex_x = numpy.clip(vertex_x, 0, max_shape[1] - 1)
            vertex_y = numpy.clip(vertex_y, 0, max_shape[0] - 1)
        partial_vertices = numpy.vstack(
            (vertex_x, vertex_y))  # 2 x num_vertices
        vertices[:, :partial_vertices.shape[1]] = partial_vertices
        return vertices

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'num_ray={self.num_ray})')
        return repr_str
