"""Classes for efficient reading of multipage tif files.

PIL/pillow will parse a tif file by making a large number of very small reads. While this is in general perfectly fine
when reading from the local filesystem, in the case where a single read is turned into an http GET request (for
instance, reading from AWS S3), this can get pretty inefficient.
"""
from collections import namedtuple
import ctypes
import os
import struct
import operator
import sys


class TiffFormatError(ValueError):
    """Exception thrown when the file being read does not appear to conform to the TIF spec.
    """
    pass


class TiffParser(object):
    """Encapsulates file access in parsing a tiff file.

    Two main uses:
    1.  Populates TiffData object while first reading through pages of a multipage tif. Primary methods here are
        parseFileHeader() and parseNextImageFileDirectory().
    2.  Generates TiffIFDData object corresponding to the desired page of a multipage tif, via getOffsetDataForIFD().
    """
    INIT_IFD_SIZE = 6 + 12 * 24  # 2b num entries, N*12b entries, 4b offset to next IFD

    def __init__(self, fp, debug=True):
        """
        Parameters
        ----------
        fp: file or file-like object, open for reading
            The parser will interpret this handle as pointing to a TIFF file.
        debug: boolean, default true
            If true, the parser will keep track of the number of seeks and total bytes read in the attributes
            self.nseeks and self.bytes_read.
        """
        self._fp = fp
        self._debug = debug
        self.max_ifd_size = TiffParser.INIT_IFD_SIZE
        self._order = None
        #self.file_header = None
        #self.ifds = []
        if self._debug:
            self.nseeks = 0
            self.bytes_read = 0

    def __seek(self, pos):
        cp = self._fp.tell()
        if cp != pos:
            self._fp.seek(pos, os.SEEK_SET)
            if self._debug:
                self.nseeks += 1

    def __read(self, size=-1):
        buf = self._fp.read(size)
        if size >= 0 and len(buf) < size:
            print "incomplete read: requested %d bytes, got %d" % (size, len(buf))
        if self._debug:
            self.bytes_read += len(buf)
        return buf

    @property
    def order(self):
        """Byte order used to interpret wrapped tif file, either '<' (little-endian) or '>' (big-endian)
        """
        return self._order

    def parseFileHeader(self, destination_tiff=None):
        """
        Reads the initial 8-byte file header from the wrapped file pointer.

        Parameters:
        -----------
        destination_tiff: TiffData object, or None
            If destination_tiff is not None, then the parsed file header will be attached to the passed destination_tiff
            object as its file_header attribute, in addition to being returned from the method call.

        Returns:
        --------
        TiffFileHeader object
        """
        self.__seek(0)
        header_buf = self.__read(8)
        file_header = TiffFileHeader.fromBytes(header_buf)
        self._order = file_header.byte_order
        if destination_tiff:
            destination_tiff.file_header = file_header
        return file_header

    def parseNextImageFileDirectory(self, destination_tiff=None, ifd_offset=None):
        """
        Reads the next Image File Directory in the wrapped file.

        The offset of the next IFD within the file is determined either from the passed destination_tiff, or is passed
        explicitly in ifd_offset. One or the other must be passed.

        Parameters:
        -----------
        destination_tiff: TiffData object with a valid file_header attribute, or None
            If passed, the offset of the next IFD will be found either from the previous IFDs stored within
            destination_tiff if any, or from destination_tiff.file_header if not. The parsed IFD will be added to
            the destination_tiff.ifds sequence.

        ifd_offset: positive integer offset within the wrapped file, or None
            If destination_tiff is None and ifd_offset is passed, then ifd_offset will be used as the file offset
            at which to look for the next IFD.

        Returns:
        --------
        TiffImageFileDirectory object
        """
        if (not destination_tiff) and (ifd_offset is None):
            raise ValueError("Either destination_tiff or ifd_offset must be specified")
        if destination_tiff:
            offset = destination_tiff.ifds[-1].ifd_offset if destination_tiff.ifds else \
                destination_tiff.file_header.ifd_offset
            if not offset:
                return None
        else:
            offset = ifd_offset

        # read out our current best guess at the IFD size for this file in bytes:
        self.__seek(offset)
        ifd_buf = self.__read(self.max_ifd_size)
        # check whether we actually got enough:
        reqd_buf_size = TiffImageFileDirectory.parseIFDBufferSize(ifd_buf, self.order)
        if reqd_buf_size > self.max_ifd_size:
            self.max_ifd_size = reqd_buf_size
        if reqd_buf_size > len(ifd_buf):
            # we hope we get the full buffer on the second attempt
            ifd_buf = self.__read(reqd_buf_size)
            if len(ifd_buf) < reqd_buf_size:
                raise IOError("Unable to read all %d bytes of tiff image file directory; got only %d bytes" %
                              (reqd_buf_size, len(ifd_buf)))

        ifd = TiffImageFileDirectory.fromBytes(ifd_buf, self.order)
        if destination_tiff:
            destination_tiff.ifds.append(ifd)
        return ifd

    def getOffsetDataForIFD(self, ifd, max_buf=10**6, max_gap=1024):
        """Loads TIF tag offset and image data for the page described in the passed IFD.

        This method will typically be called from packSinglePage() rather than being used directly by clients.

        Parameters:
        -----------
        ifd: TiffImageFileDirectory

        max_buf: positive integer, default 10^6 (1MB)
            Requests a largest size to use for file reads. Multiple contiguous image strips (or other data) of less
            than max_buf in size will be read in a single read() call. If a single strip is larger than max_buf, then
            it will still be read, in a single read call requesting exactly the strip size.

        max_gap: positive integer, default 1024 (1KB)
            Specifies the largest gap in meaningful data to tolerate within a single read() call. If two items of offset
            data for a single IFD are separated by more than max_gap of data not within the IFD, then they will be read
            in multiple read() calls. If they are separated by max_gap or less, then a single read() will be used and
            the irrelevant data in between simply ignored.

        Returns:
        --------
        TiffIFDData
        """
        return_data = TiffIFDData()
        return_data.ifd = ifd

        startlengths = ifd.getOffsetStartsAndLengths()
        buf_startlens = calcReadsForOffsets(startlengths, max_buf, max_gap)

        buffers = []
        for bs, bl in buf_startlens:
            self.__seek(bs)
            buf = self.__read(bl)
            buffers.append(TiffBuffer(bs, buf))

        for entry in ifd.entries:
            if entry.isoffset:
                offset, val_length = entry.getOffsetStartAndLength()
                found = False
                for tif_buff in buffers:
                    if tif_buff.contains(offset, val_length):
                        #print "Buffer at orig offset %d, length %d, contains offset data starting at %d, length %d" % \
                        # (tif_buff.orig_offset, len(tif_buff.buffer), offset, val_length)
                        found = True
                        fmt = self.order + entry.getOffsetDataFormat()
                        vals = tif_buff.unpackFrom(fmt, offset)
                        return_data.entries_and_offsetdata.append(
                            TiffIFDEntryAndOffsetData(*(entry, vals)))
                        break
                if not found:
                    raise ValueError("Offset data at start: %d length: %d not found in available buffers" %
                                     (offset, val_length))
            else:
                return_data.entries_and_offsetdata.append(
                    TiffIFDEntryAndOffsetData(*(entry, None)))

        del buffers
        image_offsets = None
        image_bytesizes = None
        for ifd_entry_and_data in return_data.entries_and_offsetdata:
            if ifd_entry_and_data.entry.isImageDataOffsetEntry():
                if image_offsets:
                    raise TiffFormatError("Found duplicate image data offset entries in single IFD")
                image_offsets = ifd_entry_and_data.offset_data
            elif ifd_entry_and_data.entry.isImageDataByteCountEntry():
                if image_bytesizes:
                    raise TiffFormatError("Found duplicate image data byte size entries in single IFD")
                image_bytesizes = ifd_entry_and_data.offset_data

        if (not image_offsets) or (not image_bytesizes):
            raise TiffFormatError("Missing image offset or byte size data in IFD")
        if len(image_offsets) != len(image_bytesizes):
            raise TiffFormatError("Unequal numbers of image data offset and byte size entries in IFD " +
                                  "(offsets: %d, byte sizes: %d" % (len(image_offsets), len(image_bytesizes)))

        startlengths = zip(image_offsets, image_bytesizes)
        del image_offsets, image_bytesizes
        buf_startlens = calcReadsForOffsets(startlengths, max_buf, max_gap)

        buffers = []
        for bs, bl in buf_startlens:
            self.__seek(bs)
            buf = self.__read(bl)
            buffers.append(TiffBuffer(bs, buf))

        # validate that all data was read successfully and set up views
        data_views = []
        for st, l in startlengths:
            found = False
            for buf in buffers:
                if buf.contains(st, l):
                    #print "Buffer at orig offset %d, length %d, contains strip starting at %d, length %d" % \
                    #      (buf.orig_offset, len(buf.buffer), st, l)
                    data_views.append(buf.bufferFrom(st, l))
                    found = True
                    break
            if not found:
                raise TiffFormatError("Could not find buffer with data at offset: %d, size: %d" % (st, l))

        return_data.imagedata_buffers = data_views
        return return_data


def packSinglePage(parser, tiff_data=None, page_num=0):
    """Creates a string buffer with valid tif file data from a single page of a multipage tif.

    The resulting string buffer can be written to disk as a TIF file or loaded directly by PIL or similar.

    Parameters:
    -----------
    parser: TifParser object.
        The parser should be initialized with a file handle of the multipage tif from which a page is to be extracted.

    tiff_data: TiffData object, or none.
        If tiff_data is passed, the tif file header and IFDs will be read out from it. If an empty tiff_data object or
        one without all IFDs in place is passed, then the file header and remaining required IFDs will be placed into
        it. If tiff_data is None, a new TiffData object will be generated internally to the function and discarded when
        the functional call completes. Passing tiff_data basically amounts to an optimization, to prevent rereading
        data that presumably has already been parsed out from the file.

    page_num: nonnegative integer page number
        Specifies the zero-based page number to be read out and repacked from the multipage tif wrapped by the passed
        parser object.

    Returns:
    --------
    string of bytes, making up a valid single-page TIF file.
    """
    if not tiff_data:
        tiff_data = TiffData()
    if not tiff_data.file_header:
        parser.parseFileHeader(destination_tiff=tiff_data)
    while len(tiff_data.ifds) < page_num:
        parser.parseNextImageFileDirectory(destination_tiff=tiff_data)

    tif_ifd_data = parser.getOffsetDataForIFD(tiff_data.ifds[page_num])
    order = parser.order

    preamble = TiffFileHeader.new(order, tiff_data.file_header.byteSize())
    buf_size = preamble.byteSize() + tif_ifd_data.byteSize()

    out_buffer = ctypes.create_string_buffer(buf_size)
    offset = preamble.toBytes(out_buffer, 0)
    ifd_size = tif_ifd_data.ifd.byteSize()
    ifd_data_offset = offset + ifd_size
    img_data_offset = ifd_data_offset + tif_ifd_data.ifd.getTotalOffsetSize()
    # write IFD
    offset += tif_ifd_data.ifd.toBytes(ifd_data_offset, out_buffer, dest_offset=offset, order=order)
    # write offset IFD values
    for entry, value in tif_ifd_data.entries_and_offsetdata:
        if entry.isoffset:
            fmt = order + entry.getOffsetDataFormat()
            l = struct.calcsize(fmt)
            if entry.isImageDataOffsetEntry():
                # reset image data offsets
                min_orig_offset = reduce(min, value)
                value = [v - min_orig_offset + img_data_offset for v in value]
            struct.pack_into(fmt, out_buffer, offset, *value)
            offset += l
    # write image data values
    # assert offset == img_data_offset
    for img_buf in tif_ifd_data.imagedata_buffers:
        out_buffer[offset:(offset+len(img_buf))] = img_buf
        offset += len(img_buf)
    return out_buffer.raw


class TiffBuffer(object):
    """Utility object to hold results of file read() calls.
    """
    def __init__(self, orig_offset, buffer):
        self.orig_offset = orig_offset
        self.buffer = buffer
        self.buffer_len = len(buffer)

    def contains(self, offset, length):
        lbuf = len(self.buffer)
        start_inbounds = offset >= self.orig_offset
        end_inbounds = offset + length <= self.orig_offset + lbuf
        return start_inbounds and end_inbounds
        #return offset >= self.orig_offset and length <= len(self.buffer)

    def unpackFrom(self, fmt, orig_offset):
        """Deserializes data within this buffer according to the passed format, which will be interpreted by the
        python struct package.

        Returns tuple of values. (May be a one-tuple.)
        """
        return struct.unpack_from(fmt, self.buffer, offset=orig_offset-self.orig_offset)

    def bufferFrom(self, orig_offset, size=-1):
        if size < 0:
            return buffer(self.buffer, orig_offset-self.orig_offset)
        else:
            return buffer(self.buffer, orig_offset-self.orig_offset, size)


class TiffData(object):
    """Minimal data structure holding a TiffFileHeader and a sequence of TiffImageFileDirectories.

    This object represents data read on an initial pass through a multipage tif's set of directories. It does not
    hold tag offset data or image data.
    """
    def __init__(self):
        self.file_header = None
        self.ifds = []


class TiffIFDData(object):
    """Data structure holding tag offset data and image data for a single tiff IFD.
    """
    def __init__(self):
        self.ifd = None
        self.entries_and_offsetdata = []
        self.imagedata_buffers = []

    def byteSize(self):
        imgdat_size = sum(len(buf) for buf in self.imagedata_buffers)
        return self.ifd.byteSize() + self.ifd.getTotalOffsetSize() + imgdat_size


class TiffFileHeader(namedtuple('_TiffFileHeader', 'byte_order magic ifd_offset')):
    """Data structure representing the 8-byte header found at the beginning of a tiff file.
    """
    @classmethod
    def fromBytes(cls, buf):
        order = buf[:2]
        if order == "II":
            code = '<'
            fmt = '<HI'
        elif order == "MM":
            code = '>'
            fmt = '>HI'
        else:
            raise TiffFormatError("Found byte order string '%s', should be 'II' or 'MM'" % order)
        magic, offset = struct.unpack_from(fmt, buf, offset=2)
        if magic != 42:
            raise TiffFormatError("Found bad magic number %d, should be 42" % magic)
        return cls(code, magic, offset)

    def toBytes(self, dest_buf, dest_offset=0):
        order_flag = "II" if self.byte_order == '<' else "MM"
        dest_buf[dest_offset:(dest_offset+2)] = order_flag
        struct.pack_into(self.byte_order+"HI", dest_buf, dest_offset+2, self.magic, self.ifd_offset)
        return self.byteSize()

    def asBytes(self):
        buf = ctypes.create_string_buffer(self.byteSize())
        self.toBytes(buf)
        return buf.raw

    def byteSize(self):
        return 8

    @classmethod
    def new(cls, order="=", offset=8):
        if order in ("=", "@"):
            order = ">" if sys.byteorder == "big" else "<"
        if not order in (">", "<"):
            raise ValueError("Order must be '>' or '<' for big or little-endian respectively; got '%s'" % order)
        return cls(order, 42, offset)


class TiffImageFileDirectory(object):
    """Data structure representing a single Image File Directory within a tif file.

    This object does not hold data stored in an offset from the IFD or image data.

    Individual IFD entries are represented within the 'entries' sequence attribute, which holds multiple
    TiffIFDEntry objects.
    """
    __slots__ = ('num_entries', 'entries', 'ifd_offset', 'order')

    @classmethod
    def parseIFDBufferSize(cls, buf, order, offset=0):
        """Returns total size of an image file directory in bytes.

        buf[offset:] is assumed to point to the beginning of an IFD.
        """
        return 6 + 12*(struct.unpack_from(order+"H", buf, offset)[0])

    @classmethod
    def fromBytes(cls, buf, order, offset=0):
        ifd = TiffImageFileDirectory()
        ifd.order = order
        ifd.num_entries = struct.unpack_from(order+"H", buf, offset)[0]
        ifd.ifd_offset = struct.unpack_from(order+"I", buf, offset + 2 + 12*ifd.num_entries)[0]
        for ientry in xrange(ifd.num_entries):
            ifd.entries.append(TiffIFDEntry.fromBytes(buf, order, offset + 2 + 12*ientry))
        return ifd

    def __init__(self):
        self.num_entries = 0
        self.entries = []
        self.ifd_offset = 0
        self.order = '='

    def byteSize(self):
        return 6+12*self.num_entries

    def getEntryValue(self, tag):
        """
        Returns value of the passed TIF tag, if present in the IFD.

        Throws IndexError if the tag is not found in the IFD.
        Throws TifFormatError if the tag is present, but the value is stored in an offset rather than in the IFD itself.
        """
        for entry in self.entries:
            # could optimize this by exiting early if entry.tag > tag, because entries should be in sorted order
            # according to the TIFF spec, but I'm not sure it's worth it.
            if entry.tag == tag:
                if entry.isoffset:
                    raise TiffFormatError("Tag %d is present, but is stored at offset %d rather than within IFD" %
                                          (tag, entry.val))
                return entry.val
        raise IndexError("Tag %d not found in IFD" % tag)

    def getImageWidth(self):
        return self.getEntryValue(IMAGE_WIDTH_TAG)

    def getImageHeight(self):
        return self.getEntryValue(IMAGE_HEIGHT_TAG)

    def getBitsPerSample(self):
        return self.getEntryValue(BITS_PER_SAMPLE_TAG)

    def getSampleFormat(self):
        try:
            sample_format = self.getEntryValue(SAMPLE_FORMAT_TAG)
        except KeyError:
            # default according to tif spec is UINT
            sample_format = SAMPLE_FORMAT_UINT
        return sample_format

    def getOffsetStartsAndLengths(self):
        startlengths = [entry.getOffsetStartAndLength() for entry in self.entries]
        startlengths = filter(None, startlengths)
        return startlengths

    def getTotalOffsetSize(self):
        startlengths = self.getOffsetStartsAndLengths()
        return sum(sl[1] for sl in startlengths)

    def hasEntry(self, tag):
        found = False
        for entry in self.entries:
            if entry.tag == tag:
                found = True
                break
        return found

    def isLuminanceImage(self):
        try:
            interp = self.getEntryValue(PHOTOMETRIC_INTERPRETATION_TAG)
            interp_ok = interp == 0 or interp == 1  # min is white or min is black
        except IndexError:
            # if we are missing the photometric interpretation tag, even though technically it's required,
            # check that samples per pixel is either absent or 1
            interp_ok = (not self.hasEntry(SAMPLES_PER_PIXEL_TAG)) or (self.getEntryValue(SAMPLES_PER_PIXEL_TAG) == 1)
        return interp_ok

    def __str__(self):
        entries = [str(entry) for entry in self.entries]
        return "Image File Directory\nnumber fields: %d\n%s\nnext IFD offset: %d" % \
               (self.num_entries, '\n'.join(entries), self.ifd_offset)

    def toBytes(self, new_offset, dest_buf, dest_offset=0, order="="):
        orig_dest_offset = dest_offset
        struct.pack_into(order+"H", dest_buf, dest_offset, self.num_entries)
        dest_offset += 2
        for entry in self.entries:
            if entry.isoffset:
                st, l = entry.getOffsetStartAndLength()
                dest_offset += entry.toBytes(dest_buf, new_offset=new_offset, dest_offset=dest_offset, order=order)
                new_offset += l
            else:
                dest_offset += entry.toBytes(dest_buf, dest_offset=dest_offset, order=order)
        # write "last IFD" marker:
        struct.pack_into(order+"I", dest_buf, dest_offset, 0)
        dest_offset += 4
        return dest_offset - orig_dest_offset


class TiffIFDEntry(namedtuple('_TiffIFDEntry', 'tag type count val isoffset')):
    """Data structure representing a single entry within a tif IFD.

    Data stored in an offset from the IFD table will not be explicitly represented in this object; only the file offset
    itself will be stored.
    """
    @classmethod
    def fromBytes(cls, buf, order, offset=0):
        tag, type, count = struct.unpack_from(order+'HHI', buf, offset)
        rawval = buf[(offset+8):(offset+12)]

        tagtype = IFD_ENTRY_TYPECODE_TO_TAGTYPE[type]
        bytesize = count * tagtype.size
        isoffset = bytesize > 4 or tagtype.type == 'UNK'
        if not isoffset:
            val = struct.unpack_from(order+tagtype.fmt*count, rawval)
            if count == 1:
                val = val[0]
        else:
            val = struct.unpack(order+'I', rawval)[0]
        return cls(tag, type, count, val, isoffset)

    def toBytes(self, dest_buf, new_offset=None, dest_offset=0, order="="):
        if new_offset is None and self.isoffset:
            new_offset = self.val

        if self.isoffset:
            val_fmt = 'L'
            val = new_offset
        else:
            val_fmt = lookup_tagtype(self.type).fmt * self.count
            val = self.val

        fmt = order+"HHI"+val_fmt
        fmt_size = struct.calcsize(fmt)
        if fmt_size < 12:
            fmt += 'x'*(12-fmt_size)

        packing = [self.tag, self.type, self.count]
        if isinstance(val, tuple):
            packing += list(val)
        else:
            packing.append(val)

        struct.pack_into(fmt, dest_buf, dest_offset, *packing)
        return 12

    def asBytes(self, new_offset=None, order="="):
        buf = ctypes.create_string_buffer(self.byteSize())
        self.toBytes(buf, new_offset=new_offset, dest_offset=0, order=order)
        return buf.raw

    def byteSize(self):
        return 12

    def getOffsetStartAndLength(self):
        if not self.isoffset:
            return None
        tagtype = IFD_ENTRY_TYPECODE_TO_TAGTYPE[self.type].fmt
        l = struct.calcsize("=" + tagtype * self.count)
        return self.val, l

    def getOffsetDataFormat(self):
        return IFD_ENTRY_TYPECODE_TO_TAGTYPE[self.type].fmt * self.count

    def isImageDataOffsetEntry(self):
        return self.tag in IMAGE_DATA_OFFSET_TAGS

    def isImageDataByteCountEntry(self):
        return self.tag in IMAGE_DATA_BYTECOUNT_TAGS

    def __str__(self):
        tagname = TAG_TO_NAME.get(self.tag, 'UNK')
        typename = lookup_tagtype(self.type).type
        return "TiffIFDEntry(tag: %s (%d), type: %s (%d), count=%d, val=%s%s)" % \
               (tagname, self.tag, typename, self.type, self.count, self.val,
                ' (offset)' if self.isoffset else '')


class TiffIFDEntryAndOffsetData(namedtuple("_TiffIFDEntryAndOffsetData", "entry offset_data")):
    """Simple pair structure to hold a TiffIFDEntry and its associated offset data, if any.

    If offset data is present (entry.isoffset is True), then it will be stored in offset_data as a tuple. If no
    offset data is present, then offset_data will be None.
    """
    pass


def lookup_tagtype(typecode):
    return IFD_ENTRY_TYPECODE_TO_TAGTYPE.get(typecode, UNKNOWN_TAGTYPE)


def calcReadsForOffsets(startLengthPairs, max_buf=10**6, max_gap=1024):
    """Plans a sequence of file reads and seeks to cover all the spans of data in startLengthPairs.

    Parameters:
    -----------
    startLengthPairs: sequence of (int start, int length) pairs
        start is the offset position of an item of data, length is its size in bytes.

    max_buf: positive integer, default 10^6 (1MB)
            Requests a largest size to use for file reads. Multiple contiguous image strips (or other data) of less
            than max_buf in size will be read in a single read() call. If a single strip is larger than max_buf, then
            it will still be read, in a single read call requesting exactly the strip size.

    max_gap: positive integer, default 1024 (1KB)
            Specifies the largest gap in meaningful data to tolerate within a single read() call. If two items of offset
            data for a single IFD are separated by more than max_gap of data not within the IFD, then they will be read
            in multiple read() calls. If they are separated by max_gap or less, then a single read() will be used and
            the irrelevant data in between simply ignored.

    Returns:
    --------
    sequence of (start, length) pairs, each representing a planned file read

    """
    # sort by starting position
    # we assume here that starts and offsets and generally sane - meaning (roughly) nonoverlapping
    startlengths = sorted(startLengthPairs, key=operator.itemgetter(0))

    bufstarts = []
    buflens = []
    curstart, curlen = startlengths.pop(0)
    for start, length in startlengths:
        gap = start - (curstart + curlen)
        newlen = start + length - curstart
        if gap > max_gap or newlen > max_buf:
            bufstarts.append(curstart)
            buflens.append(curlen)
            curstart = start
            curlen = length
        else:
            curlen = newlen
    bufstarts.append(curstart)
    buflens.append(curlen)
    return zip(bufstarts, buflens)

TiffTagType = namedtuple('TiffTagType', 'code type fmt size')

TAG_TO_NAME = {
    254: 'NewSubfileType',
    255: 'SubfileType',
    256: 'ImageWidth',
    257: 'ImageLength',
    258: 'BitsPerSample',
    259: 'Compression',
    262: 'PhotometricInterpretation',
    266: 'FillOrder',
    269: 'DocumentName',
    273: 'StripOffsets',
    274: 'Orientation',
    277: 'SamplesPerPixel',  # e.g. 3 for RGB images
    278: 'RowsPerStrip',
    279: 'StripByteCounts',
    282: 'XResolution',
    283: 'YResolution',
    284: 'PlanarConfiguration',
    296: 'ResolutionUnit',
    297: 'PageNumber',
    317: 'Predictor',
    318: 'WhitePoint',
    319: 'PrimaryChromaticities',
    320: 'ColorMap',
    322: 'TileWidth',
    323: 'TileLength',
    324: 'TileOffsets',
    325: 'TileByteCounts',
    338: 'ExtraSamples',
    339: 'SampleFormat'
}

IMAGE_WIDTH_TAG = 256
IMAGE_HEIGHT_TAG = 257
BITS_PER_SAMPLE_TAG = 258
PHOTOMETRIC_INTERPRETATION_TAG = 262
SAMPLES_PER_PIXEL_TAG = 277
SAMPLE_FORMAT_TAG = 339

IMAGE_DATA_OFFSET_TAGS = frozenset([273, 324])
IMAGE_DATA_BYTECOUNT_TAGS = frozenset([279, 325])

SAMPLE_FORMAT_UINT = 1
SAMPLE_FORMAT_INT = 2
SAMPLE_FORMAT_FLOAT = 3
SAMPLE_FORMAT_UNKNOWN = 4

UNKNOWN_TAGTYPE = TiffTagType(-1, 'UNK', 'L', 4)

IFD_ENTRY_TYPECODE_TO_TAGTYPE = {
    1: TiffTagType(1, 'BYTE', 'B', 1),
    2: TiffTagType(2, 'ASCII', 'c', 1),
    3: TiffTagType(3, 'SHORT', 'H', 2),
    4: TiffTagType(4, 'LONG', 'L', 4),
    5: TiffTagType(5, 'RATIONAL', 'LL', 8),
    6: TiffTagType(6, 'SBYTE', 'b', 1),
    7: TiffTagType(7, 'UNDEFINED', 'c', 1),
    8: TiffTagType(8, 'SSHORT', 'h', 2),
    9: TiffTagType(9, 'SLONG', 'l', 4),
    10: TiffTagType(10, 'SRATIONAL', 'll', 8),
    11: TiffTagType(11, 'FLOAT', 'f', 4),
    12: TiffTagType(12, 'DOUBLE', 'd', 8)
}