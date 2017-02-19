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

import warnings
from os.path import join as pjoin, dirname
import numpy as np
import zlib
import re
from operator import mul
from functools import reduce
from .keywordonly import kw_only_meth
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file
from .fileholders import FileHolder
from .py3k import asstr
from .affines import dot_reduce

class MetaImageError(Exception):
    """
    Exception for MetaImage format related problems.
    """

class MetaImageHeader(SpatialHeader):
    """MetaImage format header"""
    delim = '='
    avail_types = {
        'MET_FLOAT' : 'f4',
        'MET_DOUBLE' : 'f8',
        'MET_CHAR' : 'i1',
        'MET_UCHAR' : 'u1',
        'MET_SHORT' : 'i2',
        'MET_USHORT' : 'u2',
        'MET_INT' : 'i4',
        'MET_UINT' : 'u4'
    }
    
    def __init__(self, elementType=np.float32, dimSize=(0,), elementSpacing=(1.,),
                 elementNumberOfChannels=1, pixelDataOffset=0, extras={}):
        """
        Parameters
        ----------
        elementType : NumPy data type
            Data type of pixel data
        dimSize : tuple, int
            Shape of image matrix
        elementSpacing : tuple, float
            Physical dimensions of pixel/voxel
        elementNumberOfChannels : int
            Number of values associated with any pixel/voxel.  This is
            typically 1, although with RGB data, for instance, 3
            values (red,green,blue) corresponds to 3 channels.
        extras : dict
            key/value pairs of additional meta-information associated
            with the data
        """
        self._numChannels = elementNumberOfChannels
        self._pixelDataOffset = pixelDataOffset
        self._extras = extras

        assert len(dimSize) == len(elementSpacing)
        shape_and_channels = dimSize
        spacing_and_channel = elementSpacing
        if elementNumberOfChannels > 1:
            shape_and_channels = tuple(list(dimSize) + [elementNumberOfChannels])
            spacing_and_channel = tuple(list(elementSpacing) + [1.])
        super(MetaImageHeader, self).__init__(data_dtype=elementType,
                                              shape=shape_and_channels,
                                              zooms=spacing_and_channel)

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            return klass()
        toRet = klass(header.get_data_dtype(), header.get_data_shape(),
                      header.get_zooms(), 1,
                      header._extras)
        toRet._numChannels = header._numChannels
        return toRet

    @classmethod
    def from_fileobj(klass, fileobj):
        """ Generate class object from provided file object `fileobj`

        Parameters
        ----------
        fileobj : python native file object
            Contains the file to intialize the data from
        """
        # parse header keyvalue pairs
        hdrmap = {}
        imgoffset = 0
        for line in fileobj:
            # Separate into key,value pair
            lineparts = asstr(line).split(MetaImageHeader.delim)
            if len(lineparts) < 2:
                continue
            linekey = lineparts[0].strip()
            linevalue = MetaImageHeader.delim.join(lineparts[1:]).strip()
            hdrmap[linekey] = linevalue

            # Store header size if available
            if linekey == 'HeaderSize':
                imgoffset = int(linevalue)
            
            if linekey == 'ElementDataFile' and linevalue == 'LOCAL':
                # Infer pixel data offset, if possible
                if 'HeaderSize' not in hdrmap:
                    imgoffset = fileobj.tell()
                break
            
        # Check for unsupported data
        if hdrmap.get('ObjectType', 'Image') != 'Image':
            raise MetaImageError("Unsupported object type %s" % hdrmap['ObjectType'])

        # take out header elements with a required format
        if 'HeaderSize' in hdrmap:
            del hdrmap['HeaderSize']
        data_dtype = '<'
        if hdrmap.pop('ElementByteOrderMSB', 'False') == 'True':
            data_dtype = '>'
        data_dtype += klass.avail_types.get(hdrmap.pop('ElementType','MET_FLOAT'), np.float32)
        data_shape = klass.parse_shape(hdrmap.pop('DimSize', '0'))
        data_zooms = tuple([float(i) for i in
                            hdrmap.pop('ElementSpacing',' '.join(('1',)*len(data_shape))).split()])
        data_channels = int(hdrmap.pop('ElementNumberOfChannels','1'))

        # use constructor to generate returned object
        return klass(elementType=np.dtype(data_dtype), dimSize=data_shape,
                     elementSpacing=data_zooms,
                     elementNumberOfChannels=data_channels,
                     pixelDataOffset=imgoffset, extras=hdrmap)

    @classmethod
    def parse_shape(klass, hdrtext):
        return tuple([int(i) for i in hdrtext.split()])
    
    def copy(self):
        """ Return a deep copy of this instance """
        return from_header(self)

    def get_data_offset(self):
        """ 
        Returns header size for data, only applies to ElementDataFile=LOCAL 
        """
        return self._pixelDataOffset
        
    def set_data_offset(self, offset):
        self._extras["HeaderSize"] = offset

    def get_affine(self):
        """ Compute affine transformation into image's phsyical space

        The MetaImage format uses a combination of the physical
        offset, orientation lettering, transform matrix, and voxel
        spacing.

        """
        matsize = 4
        ndim = len(self.get_data_shape())
        ndim_orig = ndim
        if self._numChannels > 1:
            ndim_orig -= 1
        iend = min(3, ndim_orig)

        # MetaIO Physical coordinates are, by default, in LP* orientation
        default_orient = np.eye(matsize)
        default_orient[0,0] = -default_orient[0,0]
        default_orient[1,1] = -default_orient[1,1]

        # Voxel size
        zooms = self.get_zooms()
        zooms3 = np.ones((4,))
        zooms3[0:iend] = zooms[0:iend]
        spacing = np.diag(zooms3.tolist())
        
        # Orientation matrix
        orient = np.eye(matsize)
        if "TransformMatrix" in self:
            orient_str = self["TransformMatrix"].split()
            if len(orient_str) != ndim_orig*ndim_orig:
                raise MetaImageError("Transform matrix header (%s) does not match image dimensions (%d)" % (self["TransformMatrix"], ndim_orig))
            orient_hdr = np.array([float(val) for val in orient_str]).reshape((ndim_orig,ndim_orig)).transpose()
            orient[0:iend,0:iend] = orient_hdr[0:iend,0:iend]
                
        # Offset matrix
        offset_str = self.get("Offset", " ".join(["0" for i in
                                                  range(len(self.get_data_shape()))]))
        offset_hdr = [float(val) for val in offset_str.split()]
        offset = np.eye(matsize)
        offset[0:iend,3] = offset_hdr[0:iend]

        return dot_reduce(default_orient, offset, orient, spacing)

    def get(self, key, defaultvalue=None):
        """ Return value in header extras for key

        Parameters
        ----------
        key : str
            lookup string within dictionary
        defaultvalue : str (optional)
            if `key` is not in the dictionary, return this default value
        """
        if key not in self and defaultvalue is not None:
            return defaultvalue

        return self[key]
    
    def __getitem__(self, key):
        """ Return value from extras dictionary

        Equivalent to ``self._extras[key]``
        """
        return self._extras[key]

    def __contains__(self, key):
        """ True if key in extras dictionary

        Equivalent to ``key in self._extras``
        """
        return key in self._extras

    def __setitem__(self, key, value):
        """ Set item in dictionary

        Equivalent to ``self._extras[key] = value``
        """
        self._extras[key] = value

    def __delitem__(self, key):
        """ Delete key/value pair in dictionary, by key
        
        Equivalent to ``del self._extras[key]``
        """
        del self._extras[key]


class MetaImageImage(SpatialImage):
    """MetaImage format image object"""
    header_class = MetaImageHeader

    makeable = False
    rw = False

    @classmethod
    def filespec_to_file_map(klass, filespec):
        """ Make `file_map` object from filename `filespec`

        Class method

        Parameters
        ----------
        filespec : str
            Filename that might be for this image file type.

        Returns
        -------
        file_map : dict
            `file_map` dict with (key, value) pairs of (``file_type``,
            FileHolder instance), where ``file_type`` is a string
            giving the type of the contained file.  For the MetaImage
            format, a header file and image file is in order.  These files
            may or may not be the same.

        Raises
        ------
        ImageFileError
            if `filespec` is not recognizable as being a filename for this
            image type.
        """
        file_map = {}

        # Header is always the original filename, so we'll start there
        file_map['header'] = FileHolder(filename=filespec)
        
        # Image object is pointed to by the header file
        with file_map['header'].get_prepare_fileobj('rb') as hdrfile:
            volshape = None
            for line in hdrfile:
                # Separate into key,value pair
                lineparts = asstr(line).split(MetaImageHeader.delim)
                if len(lineparts) < 2:
                    continue
                linekey = lineparts[0].strip()
                linevalue = MetaImageHeader.delim.join(lineparts[1:]).strip()

                # Store shape
                if linekey == 'DimSize':
                    volshape = MetaImageHeader.parse_shape(linevalue)
                
                # Determine where pixel data is stored
                if linekey == 'ElementDataFile':
                    if linevalue == 'LOCAL':
                        file_map['image'] = FileHolder(filename=filespec, pos=hdrfile.tell())
                        break
                    elif linevalue == 'LIST':
                        if volshape is None or len(volshape) == 0:
                            raise MetaImageError('No pixel matrix shape found')
                        file_map['image'] = [FileHolder(filename=klass._relative_path_abs(asstr(hdrfile.fobj.readline().strip()), filespec)) for i in range(volshape[-1])]
                    elif klass._is_regex(asstr(linevalue)):
                        file_map['image'] = [FileHolder(filename=klass._relative_path_abs(i, filespec)) for i in klass._regex_to_list(asstr(linevalue))]
                    else:
                        file_map['image'] = FileHolder(filename=klass._relative_path_abs(linevalue, filespec))

        if 'image' not in file_map:
            raise MetaImageError('No pixel data referenced in header file')
        return file_map

    @classmethod
    def _relative_path_abs(klass, imgfilespec, fromfilespec):
        # Find file with relative paths, if necessary
        relpath = dirname(fromfilespec)
        return pjoin(relpath, imgfilespec)

    @classmethod
    def _regex(klass):
        return '(.+)\s+(\d+)\s+(\d+)\s+(\d+)'
    
    @classmethod
    def _is_regex(klass, filetext):
        return bool( re.search(klass._regex(), filetext) )

    @classmethod
    def _regex_to_list(klass, regex):
        match = re.search(klass._regex(), regex)
        toret = []
        for i in range(int(match.group(2)), int(match.group(3))+1, int(match.group(4))):
            toret.append(match.group(1) % i)
        return toret
                        
    @classmethod
    @kw_only_meth(1)
    def from_filename(klass, filename):
        """ Load MetaImage formatted file as an image

        Parameters
        ----------
        filename : str
            Filename of ".mhd" or ".mha" file to load. This file
            contains a pointer to the raw image data. In the case of a
            ".mhd" file, the name of another file containing the pixel
            data is typically used.  For a ".mha", the file itself
            typically contains the pixel data.
        """
        filemap = klass.filespec_to_file_map(filename)
        return klass.from_file_map(filemap)

    @classmethod
    @kw_only_meth(1)
    def from_file_map(klass, filemap):
        """ Load MetaImage image from a file map (see filebasedimages for more details)
        
        Parameters
        ----------
        filemap: dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective image and header files.
            N.B. image and header filenames can be the same
        """
        # Load header using header class
        with filemap['header'].get_prepare_fileobj('rb') as hdrfile:
            hdr = klass.header_class.from_fileobj(hdrfile)

        # Load image pixel data
        islist = type(filemap['image']) is list or type(filemap['image']) is tuple
        if hdr.get('CompressedData', 'False') == 'True':
            if islist:
                pixeldata = np.zeros(hdr.get_data_shape(), dtype=hdr.get_data_dtype())
                for i in range(pixeldata.shape[-1]):
                    with filemap['image'][i].get_prepare_fileobj('rb') as imgfile:
                        pixeldata[...,i] = np.ndarray(hdr.get_data_shape()[:-1],
                                                      hdr.get_data_dtype(),
                                                      buffer=zlib.decompress(imgfile.read()),
                                                      order='F')
            else:
                with filemap['image'].get_prepare_fileobj('rb') as imgfile:
                    pixeldata = np.ndarray(hdr.get_data_shape(),
                                           hdr.get_data_dtype(),
                                           buffer=zlib.decompress(imgfile.read()),
                                           order='F')
            pixeldata.flags.writeable = True
        else:
            if islist:
                pixeldata = np.zeros(hdr.get_data_shape(), dtype=hdr.get_data_dtype())
                for i in range(pixeldata.shape[-1]):
                    with filemap['image'][i].get_prepare_fileobj('rb') as imgfile:
                        pixeldata[...,i] = array_from_file(hdr.get_data_shape()[:-1],
                                                           hdr.get_data_dtype(),
                                                           imgfile,
                                                           klass._determine_pixel_offset(hdr, imgfile),
                                                           'F', False)
            else:
                with filemap['image'].get_prepare_fileobj('rb') as imgfile:
                    pixeldata = array_from_file(hdr.get_data_shape(),
                                                hdr.get_data_dtype(),
                                                imgfile,
                                                klass._determine_pixel_offset(hdr, imgfile),
                                                'F', False)
                
        return klass(pixeldata, hdr.get_affine(), header=hdr,
                     extra=None, file_map=filemap)                

    @classmethod
    def _determine_pixel_offset(klass, hdr, imgfileobj):
        # Handle case where provided header size is less than zero
        # Specify –1 to have MetaImage calculate the header size based on the assumption that the data occurs at the end of the file.
        pixeloffset = hdr.get_data_offset()
        if pixeloffset < 0:
            prevpos = imgfileobj.tell()
            imgfileobj.seek(0, 2)
            n_bytes = reduce(mul, hdr.get_data_shape()) * hdr.get_data_dtype().itemsize
            pixeloffset = imgfileobj.tell() - n_bytes
        return pixeloffset

    load = from_filename


load = MetaImageImage.load
