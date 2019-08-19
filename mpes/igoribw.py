#!/usr/bin/python
#
# Originally from W. Trevor King <wking@drexel.edu>
# Modified and adapted to the mpes project by R. Patrick Xian
#
# Based on WaveMetric's Technical Note 003, "Igor Binary Format"
#   ftp://ftp.wavemetrics.net/IgorPro/Technical_Notes/TN003.zip
# From ftp://ftp.wavemetrics.net/IgorPro/Technical_Notes/TN000.txt
#   We place no restrictions on copying Technical Notes, with the
#   exception that you cannot resell them. So read, enjoy, and
#   share. We hope IGOR Technical Notes will provide you with lots of
#   valuable information while you are developing IGOR applications.

from __future__ import print_function, division
import array, struct, types
import sys
import numpy as np


class Field (object):
    """
    Represent a Structure field.

    See Also
    --------
    Structure
    """
    def __init__(self, format, name, default=None, help=None, count=1):
        self.format = format # See the struct documentation
        self.name = name
        self.default = None
        self.help = help
        self.count = count
        self.total_count = np.prod(count)

class Structure (struct.Struct):
    """
    Represent a C structure.

    A convenient wrapper around struct.Struct that uses Fields and
    adds dict-handling methods for transparent name assignment.

    See Also
    --------
    Field

    Examples
    --------

    Represent the C structure::

        struct thing {
          short version;
          long size[3];
        }

    As

    >>> from pprint import pprint
    >>> thing = Structure(name='thing',
    ...     fields=[Field('h', 'version'), Field('l', 'size', count=3)])
    >>> thing.set_byte_order('>')
    >>> b = array.array('b', range(2+4*3))
    >>> d = thing.unpack_dict_from(buffer=b)
    >>> pprint(d)
    {'size': array([ 33752069, 101124105, 168496141]), 'version': 1}
    >>> [hex(x) for x in d['size']]
    ['0x2030405L', '0x6070809L', '0xa0b0c0dL']

    You can even get fancy with multi-dimensional arrays.

    >>> thing = Structure(name='thing',
    ...     fields=[Field('h', 'version'), Field('l', 'size', count=(3,2))])
    >>> thing.set_byte_order('>')
    >>> b = array.array('b', range(2+4*3*2))
    >>> d = thing.unpack_dict_from(buffer=b)
    >>> d['size'].shape
    (3, 2)
    >>> pprint(d)
    {'size': array([[ 33752069, 101124105],
           [168496141, 235868177],
           [303240213, 370612249]]),
     'version': 1}
    """
    def __init__(self, name, fields, byte_order='='):
        # '=' for native byte order, standard size and alignment
        # See http://docs.python.org/library/struct for details
        self.name = name
        self.fields = fields
        self.set_byte_order(byte_order)

    def __str__(self):
        return self.name

    def set_byte_order(self, byte_order):
        """Allow changing the format byte_order on the fly.
        """
        try:
            if (hasattr(self, 'format') and self.format != None
                    and self.format.startswith(byte_order)):
                return  # no need to change anything
        except:
            pass
        format = []
        for field in self.fields:
            format.extend([field.format]*field.total_count)
        struct.Struct.__init__(self, format=byte_order+''.join(format).replace('P', 'L'))

    def _flatten_args(self, args):
        # handle Field.count > 0
        flat_args = []
        for a,f in zip(args, self.fields):
            if f.total_count > 1:
                flat_args.extend(a)
            else:
                flat_args.append(a)
        return flat_args

    def _unflatten_args(self, args):
        # handle Field.count > 0
        unflat_args = []
        i = 0
        for f in self.fields:
            if f.total_count > 1:
                data = np.array(args[i:i+f.total_count])
                data = data.reshape(f.count)
                unflat_args.append(data)
            else:
                unflat_args.append(args[i])
            i += f.total_count
        return unflat_args
        
    def pack(self, *args):
        return struct.Struct.pack(self, *self._flatten_args(args))

    def pack_into(self, buffer, offset, *args):
        return struct.Struct.pack_into(self, buffer, offset,
                                       *self._flatten_args(args))

    def _clean_dict(self, dict):
        for f in self.fields:
            if f.name not in dict:
                if f.default != None:
                    dict[f.name] = f.default
                else:
                    raise ValueError('%s field not set for %s'
                                     % f.name, self.__class__.__name__)
        return dict

    def pack_dict(self, dict):
        dict = self._clean_dict(dict)
        return self.pack(*[dict[f.name] for f in self.fields])

    def pack_dict_into(self, buffer, offset, dict={}):
        dict = self._clean_dict(dict)
        return self.pack_into(buffer, offset,
                              *[dict[f.name] for f in self.fields])

    def unpack(self, string):
        return self._unflatten_args(struct.Struct.unpack(self, string))

    def unpack_from(self, buffer, offset=0):
        return self._unflatten_args(
            struct.Struct.unpack_from(self, buffer, offset))

    def unpack_dict(self, string):
        return dict(zip([f.name for f in self.fields],
                        self.unpack(string)))

    def unpack_dict_from(self, buffer, offset=0):
        return dict(zip([f.name for f in self.fields],
                        self.unpack_from(buffer, offset)))


# Numpy doesn't support complex integers by default, see
#   http://mail.python.org/pipermail/python-dev/2002-April/022408.html
#   http://mail.scipy.org/pipermail/numpy-discussion/2007-October/029447.html
# So we roll our own types.  See
#   http://docs.scipy.org/doc/numpy/user/basics.rec.html
#   http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html
complexInt8 = np.dtype([('real', np.int8), ('imag', np.int8)])
complexInt16 = np.dtype([('real', np.int16), ('imag', np.int16)])
complexInt32 = np.dtype([('real', np.int32), ('imag', np.int32)])
complexUInt8 = np.dtype([('real', np.uint8), ('imag', np.uint8)])
complexUInt16 = np.dtype([('real', np.uint16), ('imag', np.uint16)])
complexUInt32 = np.dtype([('real', np.uint32), ('imag', np.uint32)])


# Begin IGOR constants and typedefs from IgorBin.h

# From IgorMath.h
TYPE_TABLE = {       # (key: integer flag, value: numpy dtype)
    0:None,          # Text wave, not handled in ReadWave.c
    1:np.complex, # NT_CMPLX, makes number complex.
    2:np.float32, # NT_FP32, 32 bit fp numbers.
    3:np.complex64,
    4:np.float64, # NT_FP64, 64 bit fp numbers.
    5:np.complex128,
    8:np.int8,    # NT_I8, 8 bit signed integer. Requires Igor Pro
                     # 2.0 or later.
    9:complexInt8,
    0x10:np.int16,# NT_I16, 16 bit integer numbers. Requires Igor
                     # Pro 2.0 or later.
    0x11:complexInt16,
    0x20:np.int32,# NT_I32, 32 bit integer numbers. Requires Igor
                     # Pro 2.0 or later.
    0x21:complexInt32,
#   0x40:None,       # NT_UNSIGNED, Makes above signed integers
#                    # unsigned. Requires Igor Pro 3.0 or later.
    0x48:np.uint8,
    0x49:complexUInt8,
    0x50:np.uint16,
    0x51:complexUInt16,
    0x60:np.uint32,
    0x61:complexUInt32,
}

# From wave.h
MAXDIMS = 4

# From binary.h
BinHeaderCommon = Structure(  # WTK: this one is mine.
    name='BinHeaderCommon',
    fields=[
        Field('h', 'version', help='Version number for backwards compatibility.'),
        ])

BinHeader1 = Structure(
    name='BinHeader1',
    fields=[
        Field('h', 'version', help='Version number for backwards compatibility.'),
        Field('l', 'wfmSize', help='The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.'),
        Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        ])

BinHeader2 = Structure(
    name='BinHeader2',
    fields=[
        Field('h', 'version', help='Version number for backwards compatibility.'),
        Field('l', 'wfmSize', help='The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.'),
        Field('l', 'noteSize', help='The size of the note text.'),
        Field('l', 'pictSize', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        ])

BinHeader3 = Structure(
    name='BinHeader3',
    fields=[
        Field('h', 'version', help='Version number for backwards compatibility.'),
        Field('h', 'wfmSize', help='The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.'),
        Field('l', 'noteSize', help='The size of the note text.'),
        Field('l', 'formulaSize', help='The size of the dependency formula, if any.'),
        Field('l', 'pictSize', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        ])

BinHeader5 = Structure(
    name='BinHeader5',
    fields=[
        Field('h', 'version', help='Version number for backwards compatibility.'),
        Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        Field('l', 'wfmSize', help='The size of the WaveHeader5 data structure plus the wave data.'),
        Field('l', 'formulaSize', help='The size of the dependency formula, if any.'),
        Field('l', 'noteSize', help='The size of the note text.'),
        Field('l', 'dataEUnitsSize', help='The size of optional extended data units.'),
        Field('l', 'dimEUnitsSize', help='The size of optional extended dimension units.', count=MAXDIMS),
        Field('l', 'dimLabelsSize', help='The size of optional dimension labels.', count=MAXDIMS),
        Field('l', 'sIndicesSize', help='The size of string indicies if this is a text wave.'),
        Field('l', 'optionsSize1', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('l', 'optionsSize2', default=0, help='Reserved. Write zero. Ignore on read.'),
        ])


# From wave.h
MAX_WAVE_NAME2 = 18 # Maximum length of wave name in version 1 and 2
                    # files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31 # Maximum length of wave name in version 5
                    # files. Does not include the trailing null.
MAX_UNIT_CHARS = 3

# Header to an array of waveform data.

WaveHeader2 = Structure(
    name='WaveHeader2',
    fields=[
        Field('h', 'type', help='See types (e.g. NT_FP64) above. Zero for text waves.'),
        Field('P', 'next', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('c', 'bname', help='Name of wave plus trailing null.', count=MAX_WAVE_NAME2+2),
        Field('h', 'whVersion', default=0, help='Write 0. Ignore on read.'),
        Field('h', 'srcFldr', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('P', 'fileName', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('c', 'dataUnits', default=0, help='Natural data units go here - null if none.', count=MAX_UNIT_CHARS+1),
        Field('c', 'xUnits', default=0, help='Natural x-axis units go here - null if none.', count=MAX_UNIT_CHARS+1),
        Field('l', 'npnts', help='Number of data points in wave.'),
        Field('h', 'aModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('d', 'hsA', help='X value for point p = hsA*p + hsB'),
        Field('d', 'hsB', help='X value for point p = hsA*p + hsB'),
        Field('h', 'wModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('h', 'swModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('h', 'fsValid', help='True if full scale values have meaning.'),
        Field('d', 'topFullScale', help='The min full scale value for wave.'), # sic, 'min' should probably be 'max'
        Field('d', 'botFullScale', help='The min full scale value for wave.'),
        Field('c', 'useBits', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('c', 'kindBits', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('P', 'formula', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('l', 'depID', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('L', 'creationDate', help='DateTime of creation.  Not used in version 1 files.'),
        Field('c', 'wUnused', default=0, help='Reserved. Write zero. Ignore on read.', count=2),
        Field('L', 'modDate', help='DateTime of last modification.'),
        Field('P', 'waveNoteH', help='Used in memory only. Write zero. Ignore on read.'),
        Field('f', 'wData', help='The start of the array of waveform data.', count=4),
        ])

WaveHeader5 = Structure(
    name='WaveHeader5',
    fields=[
        Field('P', 'next', help='link to next wave in linked list.'),
        Field('L', 'creationDate', help='DateTime of creation.'),
        Field('L', 'modDate', help='DateTime of last modification.'),
        Field('l', 'npnts', help='Total number of points (multiply dimensions up to first zero).'),
        Field('h', 'type', help='See types (e.g. NT_FP64) above. Zero for text waves.'),
        Field('h', 'dLock', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('c', 'whpad1', default=0, help='Reserved. Write zero. Ignore on read.', count=6),
        Field('h', 'whVersion', default=1, help='Write 1. Ignore on read.'),
        Field('c', 'bname', help='Name of wave plus trailing null.', count=MAX_WAVE_NAME5+1),
        Field('l', 'whpad2', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('P', 'dFolder', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        # Dimensioning info. [0] == rows, [1] == cols etc
        Field('l', 'nDim', help='Number of of items in a dimension -- 0 means no data.', count=MAXDIMS),
        Field('d', 'sfA', help='Index value for element e of dimension d = sfA[d]*e + sfB[d].', count=MAXDIMS),
        Field('d', 'sfB', help='Index value for element e of dimension d = sfA[d]*e + sfB[d].', count=MAXDIMS),
        # SI units
        Field('c', 'dataUnits', default=0, help='Natural data units go here - null if none.', count=MAX_UNIT_CHARS+1),
        Field('c', 'dimUnits', default=0, help='Natural dimension units go here - null if none.', count=(MAXDIMS, MAX_UNIT_CHARS+1)),
        Field('h', 'fsValid', help='TRUE if full scale values have meaning.'),
        Field('h', 'whpad3', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('d', 'topFullScale', help='The max and max full scale value for wave'), # sic, probably "max and min"
        Field('d', 'botFullScale', help='The max and max full scale value for wave.'), # sic, probably "max and min"
        Field('P', 'dataEUnits', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('P', 'dimEUnits', default=0, help='Used in memory only. Write zero.  Ignore on read.', count=MAXDIMS),
        Field('P', 'dimLabels', default=0, help='Used in memory only. Write zero.  Ignore on read.', count=MAXDIMS),
        Field('P', 'waveNoteH', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('l', 'whUnused', default=0, help='Reserved. Write zero. Ignore on read.', count=16),
        # The following stuff is considered private to Igor.
        Field('h', 'aModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('h', 'wModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('h', 'swModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('c', 'useBits', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('c', 'kindBits', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('P', 'formula', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('l', 'depID', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('h', 'whpad4', default=0, help='Reserved. Write zero. Ignore on read.'),
        Field('h', 'srcFldr', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('P', 'fileName', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('P', 'sIndices', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        Field('f', 'wData', help='The start of the array of data.  Must be 64 bit aligned.', count=1),
        ])

# End IGOR constants and typedefs from IgorBin.h

# Begin functions from ReadWave.c

def need_to_reorder_bytes(version):
    # If the low order byte of the version field of the BinHeader
    # structure is zero then the file is from a platform that uses
    # different byte-ordering and therefore all data will need to be
    # reordered.
    return version & 0xFF == 0

def byte_order(needToReorderBytes):
    little_endian = sys.byteorder == 'little'
    if needToReorderBytes:
        little_endian = not little_endian
    if little_endian:
        return '<'  # little-endian
    return '>'  # big-endian    

def version_structs(version, byte_order):
    if version == 1:
        bin = BinHeader1
        wave = WaveHeader2
    elif version == 2:
        bin = BinHeader2
        wave = WaveHeader2
    elif version == 3:
        bin = BinHeader3
        wave = WaveHeader2
    elif version == 5:
        bin = BinHeader5
        wave = WaveHeader5
    else:
        raise ValueError('This does not appear to be a valid Igor binary wave file. The version field = %d.\n', version);
    checkSumSize = bin.size + wave.size
    if version == 5:
        checkSumSize -= 4  # Version 5 checksum does not include the wData field.
    bin.set_byte_order(byte_order)
    wave.set_byte_order(byte_order)
    return (bin, wave, checkSumSize)

def checksum(buffer, byte_order, oldcksum, numbytes):
    x = np.ndarray(
        (numbytes/2,), # 2 bytes to a short -- ignore trailing odd byte
        dtype=np.dtype(byte_order+'h'),
        buffer=buffer)
    oldcksum += x.sum()
    if oldcksum > 2**31:  # fake the C implementation's int rollover
        oldcksum %= 2**32
        if oldcksum > 2**31:
            oldcksum -= 2**31
    return oldcksum & 0xffff

# Translated from ReadWave()
def loadibw(filename, strict=True):
    if hasattr(filename, 'read'):
        f = filename  # filename is actually a stream object
    else:
        f = open(filename, 'rb')
    try:
        b = buffer(f.read(BinHeaderCommon.size))
        version = BinHeaderCommon.unpack_dict_from(b)['version']
        needToReorderBytes = need_to_reorder_bytes(version)
        byteOrder = byte_order(needToReorderBytes)
        
        if needToReorderBytes:
            BinHeaderCommon.set_byte_order(byteOrder)
            version = BinHeaderCommon.unpack_dict_from(b)['version']
        bin_struct,wave_struct,checkSumSize = version_structs(version, byteOrder)

        b = buffer(b + f.read(bin_struct.size + wave_struct.size - BinHeaderCommon.size))
        c = checksum(b, byteOrder, 0, checkSumSize)
        if c != 0:
            raise ValueError('Error in checksum - should be 0, is %d.  This does not appear to be a valid Igor binary wave file.' % c)
        bin_info = bin_struct.unpack_dict_from(b)
        wave_info = wave_struct.unpack_dict_from(b, offset=bin_struct.size)
        if wave_info['type'] == 0:
            raise NotImplementedError('Text wave')
        if version in [1,2,3]:
            tail = 16  # 16 = size of wData field in WaveHeader2 structure
            waveDataSize = bin_info['wfmSize'] - wave_struct.size
            # =  bin_info['wfmSize']-16 - (wave_struct.size - tail)
        else:
            assert version == 5, version
            tail = 4  # 4 = size of wData field in WaveHeader5 structure
            waveDataSize = bin_info['wfmSize'] - (wave_struct.size - tail)
        # dtype() wrapping to avoid np.generic and
        # getset_descriptor issues with the builtin Numpy types
        # (e.g. int32).  It has no effect on our local complex
        # integers.
        t = np.dtype(TYPE_TABLE[wave_info['type']])
        assert waveDataSize == wave_info['npnts'] * t.itemsize, \
            ('%d, %d, %d, %s' % (waveDataSize, wave_info['npnts'], t.itemsize, t))
        tail_data = array.array('f', b[-tail:])
        data_b = buffer(buffer(tail_data) + f.read(waveDataSize-tail))
        if version == 5:
            shape = [n for n in wave_info['nDim'] if n > 0]
        else:
            shape = (wave_info['npnts'],)
        data = np.ndarray(
            shape=shape,
            dtype=t.newbyteorder(byteOrder),
            buffer=data_b,
            order='F',
            )

        if version == 1:
            pass  # No post-data information
        elif version == 2:
            # Post-data info:
            #   * 16 bytes of padding
            #   * Optional wave note data
            pad_b = buffer(f.read(16))  # skip the padding
            if max(pad_b) != 0:
                if strict:
                    assert max(pad_b) == 0, pad_b
                else:
                    print(sys.stderr, 'warning: post-data padding not zero: %s.' % pad_b)
            bin_info['note'] = str(f.read(bin_info['noteSize'])).strip()
        elif version == 3:
            # Post-data info:
            #   * 16 bytes of padding
            #   * Optional wave note data
            #   * Optional wave dependency formula
            """
            Excerpted from TN003:

            A wave has a dependency formula if it has been bound by a
            statement such as "wave0 := sin(x)". In this example, the
            dependency formula is "sin(x)". The formula is stored with
            no trailing null byte.
            """
            pad_b = buffer(f.read(16))  # skip the padding
            if max(pad_b) != 0:
                if strict:
                    assert max(pad_b) == 0, pad_b
                else:
                    print(sys.stderr, 'warning: post-data padding not zero: %s.' % pad_b)
            bin_info['note'] = str(f.read(bin_info['noteSize'])).strip()
            bin_info['formula'] = str(f.read(bin_info['formulaSize'])).strip()
        elif version == 5:
            # Post-data info:
            #   * Optional wave dependency formula
            #   * Optional wave note data
            #   * Optional extended data units data
            #   * Optional extended dimension units data
            #   * Optional dimension label data
            #   * String indices used for text waves only
            """
            Excerpted from TN003:

            dataUnits - Present in versions 1, 2, 3, 5. The dataUnits
              field stores the units for the data represented by the
              wave. It is a C string terminated with a null
              character. This field supports units of 0 to 3 bytes. In
              version 1, 2 and 3 files, longer units can not be
              represented. In version 5 files, longer units can be
              stored using the optional extended data units section of
              the file.

            xUnits - Present in versions 1, 2, 3. The xUnits field
              stores the X units for a wave. It is a C string
              terminated with a null character.  This field supports
              units of 0 to 3 bytes. In version 1, 2 and 3 files,
              longer units can not be represented.

            dimUnits - Present in version 5 only. This field is an
              array of 4 strings, one for each possible wave
              dimension. Each string supports units of 0 to 3
              bytes. Longer units can be stored using the optional
              extended dimension units section of the file.
            """
            bin_info['formula'] = str(f.read(bin_info['formulaSize'])).strip()
            bin_info['note'] = str(f.read(bin_info['noteSize'])).strip()
            bin_info['dataEUnits'] = str(f.read(bin_info['dataEUnitsSize'])).strip()
            bin_info['dimEUnits'] = [
                str(f.read(size)).strip() for size in bin_info['dimEUnitsSize']]
            bin_info['dimLabels'] = []
            for size in bin_info['dimLabelsSize']:
                labels = str(f.read(size)).split(chr(0)) # split null-delimited strings
                bin_info['dimLabels'].append([L for L in labels if len(L) > 0])
            if wave_info['type'] == 0:  # text wave
                bin_info['sIndices'] = f.read(bin_info['sIndicesSize'])

    finally:
        if not hasattr(filename, 'read'):
            f.close()

    return data, bin_info, wave_info


def saveibw(filename):
    raise NotImplementedError


if __name__ == '__main__':
    """
    IBW -> ASCII conversion
    """
    import optparse
    import sys

    p = optparse.OptionParser(version=__version__)

    p.add_option('-f', '--infile', dest='infile', metavar='FILE',
                 default='-', help='Input IGOR Binary Wave (.ibw) file.')
    p.add_option('-o', '--outfile', dest='outfile', metavar='FILE',
                 default='-', help='File for ASCII output.')
    p.add_option('-v', '--verbose', dest='verbose', default=0,
                 action='count', help='Increment verbosity')
    p.add_option('-n', '--not-strict', dest='strict', default=True,
                 action='store_false', help='Attempt to parse invalid IBW files.')
    p.add_option('-t', '--test', dest='test', default=False,
                 action='store_true', help='Run internal tests and exit.')

    options,args = p.parse_args()

    if options.test == True:
        import doctest
        num_failures,num_tests = doctest.testmod(verbose=options.verbose)
        sys.exit(min(num_failures, 127))

    if len(args) > 0 and options.infile == None:
        options.infile = args[0]
    if options.infile == '-':
        options.infile = sys.stdin
    if options.outfile == '-':
        options.outfile = sys.stdout

    data,bin_info,wave_info = loadibw(options.infile, strict=options.strict)
    np.savetxt(options.outfile, data, fmt='%g', delimiter='\t')
    if options.verbose > 0:
        import pprint
        pprint.pprint(bin_info)
        pprint.pprint(wave_info)