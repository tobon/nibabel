# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Read images in Kitware's MetaImage format.

Unlike scanner-specific formats, the MetaImage format is one developed
and maintained by a software organization, Kitware.  Originally
created for their Insight Toolkit (ITK).

###############
MetaImage file format
###############
"""

from .keywordonly import kw_only_meth
from .spatialimages import SpatialHeader, SpatialImage

class MetaImageError(Exception):
    """
    Exception for MetaImage format related problems.
    """

class MetaImageHeader(SpatialHeader):
    """MetaImage format header"""

    def __init__(self):
        pass

    @classmethod
    def from_header(klass):
        pass

    @classmethod
    def from_fileobj(klass):
        pass

    def copy(self):
        pass

    def as_analyze_map(self):
        pass

    def get_water_fat_shift(self):
        pass

    def get_echo_train_length(self):
        pass

    def get_q_vectors(self):
        pass

    def get_bvals_bvecs(self):
        pass

    def get_def(self, name):
        pass

    def _get_unique_image_prop(self, name):
        pass

    def get_voxel_size(self):
        warnings.warn('Please use "get_zooms" instead of "get_voxel_size"',
                      DeprecationWarning,
                      stacklevel=2)

    def get_data_offset(self):
        pass

    def set_data_offset(self, offset):
        pass

    def get_affine(self):
        pass

    def get_data_scaling(self):
        pass

    def get_slice_orientation(self):
        pass

    def get_rec_shape(self):
        pass

    def get_sorted_slice_indices(self):
        pass

    def get_volume_labels(self):
        pass

class MetaImage(SpatialImage):
    """MetaImage format image object"""
    header_class = MetaImageHeader
    valid_exts = ('.mha', '.mhd')
    #files_types = (('image', '.raw'), ('image', '.zraw'),  ('header', '.par'))

    makeable = False
    rw = False

    @classmethod
    @kw_only_meth(1)
    def from_file_map(klass, file_map, mmap=True, permit_truncated=False,
                      scaling='dv', strict_sort=False):
        pass

    @classmethod
    @kw_only_meth(1)
    def from_filename(klass, filename, mmap=True, permit_truncated=False,
                      scaling='dv', strict_sort=False):
        pass

    load = from_filename


load = MetaImage.load
