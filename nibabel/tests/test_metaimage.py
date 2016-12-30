# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from os.path import join as pjoin, dirname
import numpy as np
from nose import SkipTest
from nose.tools import raises
from ..metaimage import MetaImageError

DATA_PATH = pjoin(dirname(__file__), 'data')

EXAMPLE_IMAGES = [
    # Example images come from Kitware's Insight Toolkit example
    # test data.
    dict(
        fname=pjoin(DATA_PATH, '113766.003.001.mha'),
        shape=(4, 4, 3),
        dtype=np.int16,
        affine=np.array([[-1,0,0,-125], [0,0,-1,99], [0,-1,0,144], [0,0,0,1]]),
        zooms=(1.01563, 1.01563, 3),
        data_summary=dict(
            min=16,
            max=240,
            mean=122.6667),
        is_proxy=False
    ),
    
    dict(
        fname=pjoin(DATA_PATH, '19771.002.001.mha'),
        shape=(4, 4, 6),
        dtype=np.int16,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1.01562, 1.01562, 1.5),
        data_summary=dict(
            min=16,
            max=240,
            mean=128),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, '3868-2-100.mha'),
        shape=(4, 4, 6),
        dtype=np.int16,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,0]]),
        zooms=(.78125, .78125, 1.),
        data_summary=dict(
            min=16,
            max=240,
            mean=128),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'BigEndian.mhd'),
        shape=(24, 24, 1),
        dtype=np.uint16,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=17264,
            mean=8548),
        is_proxy=False
    ),
    
    dict(
        fname=pjoin(DATA_PATH, 'BinarySquare3D.mhd'),
        shape=(21, 21, 21),
        dtype=np.uint8,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=255,
            mean=36.6489),
        is_proxy=False
    ),
    
    dict(
        fname=pjoin(DATA_PATH, 'BinarySquare4D.mhd'),
        shape=(21, 21, 21, 21),
        dtype=np.uint8,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1., 1.),
        data_summary=dict(
            min=0,
            max=255,
            mean=19.1970),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'LittleEndian.mhd'),
        shape=(24, 24, 1),
        dtype=np.uint16,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=17264,
            mean=8548),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'RGBTestImageCCITTFax3.mha'),
        shape=(64, 64, 3),
        dtype=np.uint8,
        affine=np.array([[-1,0,0], [0,-1,0], [0,0,1]]),
        zooms=(.352778, .352778),
        data_summary=dict(
            min=0,
            max=255,
            mean=88.2166),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'RGBTestImageCCITTFax4.mha'),
        shape=(64, 64, 3),
        dtype=np.uint8,
        affine=np.array([[-1,0,0], [0,-1,0], [0,0,1]]),
        zooms=(.352778, .352778),
        data_summary=dict(
            min=0,
            max=255,
            mean=88.2166),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'Small Ramp Volume List.mhd'),
        shape=(6, 6, 6),
        dtype=np.uint8,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=215,
            mean=107.5),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'Small Ramp Volume Reg Ex.mhd'),
        shape=(6, 6, 6),
        dtype=np.uint8,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=215,
            mean=107.5),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'SmallRampVolumeList.mhd'),
        shape=(6, 6, 6),
        dtype=np.uint8,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=215,
            mean=107.5),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'SmallRampVolumeRegEx.mhd'),
        shape=(6, 6, 6),
        dtype=np.uint8,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=215,
            mean=107.5),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'ramp.mhd'),
        shape=(30, 30, 30),
        dtype=np.uint16,
        affine=np.array([[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=26999,
            mean=13499.5),
        is_proxy=False
    ),

    dict(
        fname=pjoin(DATA_PATH, 'smallRGBA.mha'),
        shape=(40, 20, 4),
        dtype=np.uint8,
        affine=np.array([[-1,0,0], [0,-1,0], [0,0,1]]),
        zooms=(1., 1., 1.),
        data_summary=dict(
            min=0,
            max=171,
            mean=73.905),
        is_proxy=False
    )
]

@raises(MetaImageError)
def test_missing_raw():
    raise SkipTest("TODO: MetaImage loader not implemented yet")
    metaimage.load(pjoin(DATA_PATH, "MetaImageError.mhd"))
    
        
