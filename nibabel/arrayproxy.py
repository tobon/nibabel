# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Array proxy base class

The API is - at minimum:

* The object has an attribute ``shape``
* that the object returns the data array from ``np.asarray(obj)``
* that modifying no object outside ``obj`` will affect the result of
  ``np.asarray(obj)``.  Specifically, if you pass a header into the the
  __init__, then modifying the original header will not affect the result of the
  array return.
"""

from .volumeutils import BinOpener


class ArrayProxy(object):
    """
    The array proxy allows us to freeze the passed fileobj and header such that
    it returns the expected data array.

    This fairly generic implementation allows us to deal with Analyze and its
    variants, including Nifti1, and with the MGH format, apparently.

    It requires a ``header`` object with methods:
    * copy
    * get_data_shape
    * data_from_fileobj

    Other image types might need to implement their own implementation of this
    API.  See :mod:`minc` for an example.
    """
    def __init__(self, file_like, header):
        self.file_like = file_like
        self.header = header.copy()
        self._shape = header.get_data_shape()

    @property
    def shape(self):
        return self._shape

    def __array__(self):
        ''' Read of data from file '''
        with BinOpener(self.file_like) as fileobj:
            data = self.header.data_from_fileobj(fileobj)
        return data
