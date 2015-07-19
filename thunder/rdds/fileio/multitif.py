"""Classes for efficient reading of multipage tif files.

PIL/pillow will parse a tif file by making a large number of very small reads. While this is in general perfectly fine
when reading from the local filesystem, in the case where a single read is turned into an http GET request (for
instance, reading from AWS S3), this can get pretty inefficient.

These classes, primarily accessed through TiffParser, attempt to optimize access to a single page of a multipage tif
by batching read requests into a smaller number of larger reads. The resulting page data can then be repacked into a
single-page tif file format using the packSinglePage() function. This data can then be handed off to PIL, etc
to be (again...) parsed, decompressed, turned into arrays and so on.
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
        self.maxIfdSize = TiffParser.INIT_IFD_SIZE
        self._order = None
        if self._debug:
            self.nseeks = 0
            self.bytesRead = 0
            self.nreads = 0

    def __seek(self, pos):
        cp = self._fp.tell()
        if cp != pos:
            self._fp.seek(pos, os.SEEK_SET)
            if self._debug:
                self.nseeks += 1

    def __read(self, size=-1):
        curBuf = self._fp.read(size)
        if self._debug:
            self.nreads += 1
        if size >= 0 and len(curBuf) < size:
            # print "incomplete read: requested %d bytes, got %d; retrying" % (size, len(curbuf))
            size -= len(curBuf)
            buf = ' '  # init loop
            while size > 0 and len(buf) > 0:
                # keep reading while we're still getting data (no EOF) and still have data left to get
                buf = self._fp.read(size)
                # if len(buf) < size:
                #     if len(buf) > 0:
                #         print "incomplete read: requested %d bytes, got %d; retrying" % (size, len(curbuf))
                #     else:
                #         print "incomplete read: requested %d bytes, got 0 (EOF)" % size
                if self._debug:
                    self.nreads += 1
                curBuf += buf  # costly concatenation here...
                size -= len(buf)
        if self._debug:
            self.bytesRead += len(curBuf)
        return curBuf

    @property
    def order(self):
        """Byte order used to interpret wrapped tif file, either '<' (little-endian) or '>' (big-endian)
        """
        return self._order

    def parseFileHeader(self, destinationTiff=None):
        """
        Reads the initial 8-byte file header from the wrapped file pointer.

        Parameters:
        -----------
        destinationTiff: TiffData object, or None
            If destinationTiff is not None, then the parsed file header will be attached to the passed destinationTiff
            object as its fileHeader attribute, in addition to being returned from the method call.

        Returns:
        --------
        TiffFileHeader object
        """
        self.__seek(0)
        headerBuf = self.__read(8)
        fileHeader = TiffFileHeader.fromBytes(headerBuf)
        self._order = fileHeader.byteOrder
        if destinationTiff:
            destinationTiff.fileHeader = fileHeader
        return fileHeader

    def parseNextImageFileDirectory(self, destinationTiff=None, ifdOffset=None):
        """
        Reads the next Image File Directory in the wrapped file.

        The offset of the next IFD within the file is determined either from the passed destinationTiff, or is passed
        explicitly in ifdOffset. One or the other must be passed.

        Parameters:
        -----------
        destinationTiff: TiffData object with a valid fileHeader attribute, or None
            If passed, the offset of the next IFD will be found either from the previous IFDs stored within
            destinationTiff if any, or from destinationTiff.fileHeader if not. The parsed IFD will be added to
            the destinationTiff.ifds sequence.

        ifdOffset: positive integer offset within the wrapped file, or None
            If destinationTiff is None and ifdOffset is passed, then ifdOffset will be used as the file offset
            at which to look for the next IFD.

        Returns:
        --------
        TiffImageFileDirectory object
        """
        if (not destinationTiff) and (ifdOffset is None):
            raise ValueError("Either destinationTiff or ifdOffset must be specified")
        if destinationTiff:
            offset = destinationTiff.ifds[-1].ifdOffset if destinationTiff.ifds else \
                destinationTiff.fileHeader.ifdOffset
            if not offset:
                return None
        else:
            offset = ifdOffset

        # read out our current best guess at the IFD size for this file in bytes:
        self.__seek(offset)
        ifdBuf = self.__read(self.maxIfdSize)
        # check whether we actually got enough:
        reqdBufSize = TiffImageFileDirectory.parseIFDBufferSize(ifdBuf, self.order)
        if reqdBufSize > self.maxIfdSize:
            self.maxIfdSize = reqdBufSize
        if reqdBufSize > len(ifdBuf):
            # we hope we get the full buffer on the second attempt
            ifdBuf = self.__read(reqdBufSize)
            if len(ifdBuf) < reqdBufSize:
                raise IOError("Unable to read all %d bytes of tiff image file directory; got only %d bytes" %
                              (reqdBufSize, len(ifdBuf)))

        ifd = TiffImageFileDirectory.fromBytes(ifdBuf, self.order)
        if destinationTiff:
            destinationTiff.ifds.append(ifd)
        return ifd

    def getOffsetDataForIFD(self, ifd, maxBuf=10**6, maxGap=1024):
        """Loads TIF tag offset and image data for the page described in the passed IFD.

        This method will typically be called from packSinglePage() rather than being used directly by clients.

        Parameters:
        -----------
        ifd: TiffImageFileDirectory

        maxBuf: positive integer, default 10^6 (1MB)
            Requests a largest size to use for file reads. Multiple contiguous image strips (or other data) of less
            than maxBuf in size will be read in a single read() call. If a single strip is larger than maxBuf, then
            it will still be read, in a single read call requesting exactly the strip size.

        maxGap: positive integer, default 1024 (1KB)
            Specifies the largest gap in meaningful data to tolerate within a single read() call. If two items of offset
            data for a single IFD are separated by more than maxGap of data not within the IFD, then they will be read
            in multiple read() calls. If they are separated by maxGap or less, then a single read() will be used and
            the irrelevant data in between simply ignored.

        Returns:
        --------
        TiffIFDData
        """
        returnData = TiffIFDData()
        returnData.ifd = ifd

        startLengths = ifd.getOffsetStartsAndLengths()
        bufStartLens = calcReadsForOffsets(startLengths, maxBuf, maxGap)

        buffers = []
        for bs, bl in bufStartLens:
            self.__seek(bs)
            buf = self.__read(bl)
            buffers.append(TiffBuffer(bs, buf))

        for entry in ifd.entries:
            if entry.isOffset:
                offset, valLength = entry.getOffsetStartAndLength()
                found = False
                for tiffBuff in buffers:
                    if tiffBuff.contains(offset, valLength):
                        found = True
                        fmt = self.order + entry.getOffsetDataFormat()
                        vals = tiffBuff.unpackFrom(fmt, offset)
                        returnData.entriesAndOffsetData.append(
                            TiffIFDEntryAndOffsetData(*(entry, vals)))
                        break
                if not found:
                    raise ValueError("Offset data at start: %d length: %d not found in available buffers" %
                                     (offset, valLength))
            else:
                returnData.entriesAndOffsetData.append(
                    TiffIFDEntryAndOffsetData(*(entry, None)))

        del buffers
        imageOffsets = None
        imageBytesizes = None
        for ifdEntryAndData in returnData.entriesAndOffsetData:
            if ifdEntryAndData.entry.isImageDataOffsetEntry():
                if imageOffsets:
                    raise TiffFormatError("Found duplicate image data offset entries in single IFD")
                imageOffsets = ifdEntryAndData.getData()
            elif ifdEntryAndData.entry.isImageDataByteCountEntry():
                if imageBytesizes:
                    raise TiffFormatError("Found duplicate image data byte size entries in single IFD")
                imageBytesizes = ifdEntryAndData.getData()

        if (not imageOffsets) or (not imageBytesizes):
            raise TiffFormatError("Missing image offset or byte size data in IFD")
        if len(imageOffsets) != len(imageBytesizes):
            raise TiffFormatError("Unequal numbers of image data offset and byte size entries in IFD " +
                                  "(offsets: %d, byte sizes: %d" % (len(imageOffsets), len(imageBytesizes)))

        startLengths = zip(imageOffsets, imageBytesizes)
        del imageOffsets, imageBytesizes
        bufStartLens = calcReadsForOffsets(startLengths, maxBuf, maxGap)

        buffers = []
        for bs, bl in bufStartLens:
            self.__seek(bs)
            buf = self.__read(bl)
            buffers.append(TiffBuffer(bs, buf))

        # validate that all data was read successfully and set up views
        dataViews = []
        for st, l in startLengths:
            found = False
            for buf in buffers:
                if buf.contains(st, l):
                    # print "Buffer at orig offset %d, length %d, contains strip starting at %d, length %d" % \
                    #      (buf.orig_offset, len(buf.buffer), st, l)
                    dataViews.append(buf.bufferFrom(st, l))
                    found = True
                    break
            if not found:
                raise TiffFormatError("Could not find buffer with data at offset: %d, size: %d" % (st, l))

        returnData.imagedataBuffers = dataViews
        return returnData


def packSinglePage(parser, tiffData=None, pageIdx=0):
    """Creates a string buffer with valid tif file data from a single page of a multipage tif.

    The resulting string buffer can be written to disk as a TIF file or loaded directly by PIL or similar.

    Parameters:
    -----------
    parser: TifParser object.
        The parser should be initialized with a file handle of the multipage tif from which a page is to be extracted.

    tiffData: TiffData object, or none.
        If tiffData is passed, the tif file header and IFDs will be read out from it. If an empty tiffData object or
        one without all IFDs in place is passed, then the file header and remaining required IFDs will be placed into
        it. If tiffData is None, a new TiffData object will be generated internally to the function and discarded when
        the functional call completes. Passing tiffData basically amounts to an optimization, to prevent rereading
        data that presumably has already been parsed out from the file.

    pageIdx: nonnegative integer page number
        Specifies the zero-based page number to be read out and repacked from the multipage tif wrapped by the passed
        parser object.

    Returns:
    --------
    string of bytes, making up a valid single-page TIF file.
    """
    if not tiffData:
        tiffData = TiffData()
    if not tiffData.fileHeader:
        parser.parseFileHeader(destinationTiff=tiffData)
    while len(tiffData.ifds) <= pageIdx:
        parser.parseNextImageFileDirectory(destinationTiff=tiffData)

    tiffIfdData = parser.getOffsetDataForIFD(tiffData.ifds[pageIdx])
    order = parser.order

    preamble = TiffFileHeader.new(order, tiffData.fileHeader.byteSize())
    bufSize = preamble.byteSize() + tiffIfdData.byteSize()

    outBuffer = ctypes.create_string_buffer(bufSize)
    offset = preamble.toBytes(outBuffer, 0)
    ifdSize = tiffIfdData.ifd.byteSize()
    ifdDataOffset = offset + ifdSize
    imgDataOffset = ifdDataOffset + tiffIfdData.ifd.getTotalOffsetSize()
    # write IFD
    offset += tiffIfdData.ifd.toBytes(ifdDataOffset, imgDataOffset, outBuffer,
                                      destOffset=offset, order=order)
    # write offset IFD values
    for entry, value in tiffIfdData.entriesAndOffsetdata:
        if entry.isOffset:
            fmt = order + entry.getOffsetDataFormat()
            l = struct.calcsize(fmt)
            if entry.isImageDataOffsetEntry():
                # reset image data offsets
                minOrigOffset = reduce(min, value)
                value = [v - minOrigOffset + imgDataOffset for v in value]
            struct.pack_into(fmt, outBuffer, offset, *value)
            offset += l
    # write image data values
    # assert offset == img_data_offset
    for imgBuf in tiffIfdData.imagedataBuffers:
        outBuffer[offset:(offset+len(imgBuf))] = imgBuf
        offset += len(imgBuf)
    return outBuffer.raw


class TiffBuffer(object):
    """Utility object to hold results of file read() calls.
    """
    def __init__(self, origOffset, buffer_):
        self.origOffset = origOffset
        self.buffer = buffer_
        self.bufferLen = len(buffer_)

    def contains(self, offset, length):
        lbuf = len(self.buffer)
        startInbounds = offset >= self.origOffset
        endInbounds = offset + length <= self.origOffset + lbuf
        return startInbounds and endInbounds

    def unpackFrom(self, fmt, origOffset):
        """Deserializes data within this buffer according to the passed format, which will be interpreted by the
        python struct package.

        Returns tuple of values. (May be a one-tuple.)
        """
        return struct.unpack_from(fmt, self.buffer, offset=origOffset-self.origOffset)

    def bufferFrom(self, origOffset, size=-1):
        if size < 0:
            return buffer(self.buffer, origOffset-self.origOffset)
        else:
            return buffer(self.buffer, origOffset-self.origOffset, size)


class TiffData(object):
    """Minimal data structure holding a TiffFileHeader and a sequence of TiffImageFileDirectories.

    This object represents data read on an initial pass through a multipage tif's set of directories. It does not
    hold tag offset data or image data.
    """
    def __init__(self):
        self.fileHeader = None
        self.ifds = []


class TiffIFDData(object):
    """Data structure holding tag offset data and image data for a single tiff IFD.
    """
    def __init__(self):
        self.ifd = None
        self.entriesAndOffsetData = []
        self.imagedataBuffers = []

    def byteSize(self):
        imgdatSize = sum(len(buf) for buf in self.imagedataBuffers)
        return self.ifd.byteSize() + self.ifd.getTotalOffsetSize() + imgdatSize


class TiffFileHeader(namedtuple('_TiffFileHeader', 'byteOrder magic ifdOffset')):
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

    def toBytes(self, destBuf, destOffset=0):
        orderFlag = "II" if self.byteOrder == '<' else "MM"
        destBuf[destOffset:(destOffset+2)] = orderFlag
        struct.pack_into(self.byteOrder+"HI", destBuf, destOffset+2, self.magic, self.ifdOffset)
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
        if order not in (">", "<"):
            raise ValueError("Order must be '>' or '<' for big or little-endian respectively; got '%s'" % order)
        return cls(order, 42, offset)


class TiffImageFileDirectory(object):
    """Data structure representing a single Image File Directory within a tif file.

    This object does not hold data stored in an offset from the IFD or image data.

    Individual IFD entries are represented within the 'entries' sequence attribute, which holds multiple
    TiffIFDEntry objects.
    """
    __slots__ = ('numEntries', 'entries', 'ifdOffset', 'order')

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
        ifd.numEntries = struct.unpack_from(order+"H", buf, offset)[0]
        ifd.ifdOffset = struct.unpack_from(order+"I", buf, offset + 2 + 12*ifd.numEntries)[0]
        for ientry in xrange(ifd.numEntries):
            ifd.entries.append(TiffIFDEntry.fromBytes(buf, order, offset + 2 + 12*ientry))
        return ifd

    def __init__(self):
        self.numEntries = 0
        self.entries = []
        self.ifdOffset = 0
        self.order = '='

    def byteSize(self):
        return 6+12*self.numEntries

    def getEntryValue(self, tag):
        """
        Returns value of the passed TIF tag, if present in the IFD.

        Throws KeyError if the tag is not found in the IFD.
        Throws TifFormatError if the tag is present, but the value is stored in an offset rather than in the IFD itself.
        """
        for entry in self.entries:
            # could optimize this by exiting early if entry.tag > tag, because entries should be in sorted order
            # according to the TIFF spec, but I'm not sure it's worth it.
            if entry.tag == tag:
                if entry.isOffset:
                    raise TiffFormatError("Tag %d is present, but is stored at offset %d rather than within IFD" %
                                          (tag, entry.val))
                return entry.val
        raise KeyError("Tag %d not found in IFD" % tag)

    def getImageWidth(self):
        return self.getEntryValue(IMAGE_WIDTH_TAG)

    def getImageHeight(self):
        return self.getEntryValue(IMAGE_HEIGHT_TAG)

    def getBitsPerSample(self):
        return self.getEntryValue(BITS_PER_SAMPLE_TAG)

    def getSampleFormat(self):
        try:
            sampleFormat = self.getEntryValue(SAMPLE_FORMAT_TAG)
        except KeyError:
            # default according to tif spec is UINT
            sampleFormat = SAMPLE_FORMAT_UINT
        return sampleFormat

    def getOffsetStartsAndLengths(self):
        startLengths = [entry.getOffsetStartAndLength() for entry in self.entries]
        startLengths = filter(None, startLengths)
        return startLengths

    def getTotalOffsetSize(self):
        startLengths = self.getOffsetStartsAndLengths()
        return sum(sl[1] for sl in startLengths)

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
            interpOk = interp == 0 or interp == 1  # min is white or min is black
        except IndexError:
            # if we are missing the photometric interpretation tag, even though technically it's required,
            # check that samples per pixel is either absent or 1
            interpOk = (not self.hasEntry(SAMPLES_PER_PIXEL_TAG)) or (self.getEntryValue(SAMPLES_PER_PIXEL_TAG) == 1)
        return interpOk

    def __str__(self):
        entries = [str(entry) for entry in self.entries]
        return "Image File Directory\nnumber fields: %d\n%s\nnext IFD offset: %d" % \
               (self.numEntries, '\n'.join(entries), self.ifdOffset)

    def toBytes(self, newOffset, imgDataOffset, destBuf, destOffset=0, order="="):
        origDestOffset = destOffset
        struct.pack_into(order+"H", destBuf, destOffset, self.numEntries)
        destOffset += 2
        for entry in self.entries:
            if entry.isOffset:
                st, l = entry.getOffsetStartAndLength()
                destOffset += entry.toBytes(destBuf, newOffset=newOffset,
                                            imgDataOffset=imgDataOffset,
                                            destOffset=destOffset, order=order)
                newOffset += l
            else:
                destOffset += entry.toBytes(destBuf, imgDataOffset=imgDataOffset,
                                            destOffset=destOffset, order=order)
        # write "last IFD" marker:
        struct.pack_into(order+"I", destBuf, destOffset, 0)
        destOffset += 4
        return destOffset - origDestOffset


class TiffIFDEntry(namedtuple('_TiffIFDEntry', 'tag type count val isOffset')):
    """Data structure representing a single entry within a tif IFD.

    Data stored in an offset from the IFD table will not be explicitly represented in this object; only the file offset
    itself will be stored.
    """
    @classmethod
    def fromBytes(cls, buf, order, offset=0):
        tag, type_, count = struct.unpack_from(order+'HHI', buf, offset)
        rawval = buf[(offset+8):(offset+12)]

        tagType = IFD_ENTRY_TYPECODE_TO_TAGTYPE[type_]
        bytesize = count * tagType.size
        isOffset = bytesize > 4 or tagType.type == 'UNK'
        if not isOffset:
            val = struct.unpack_from(order+tagType.fmt*count, rawval)
            if count == 1:
                val = val[0]
        else:
            val = struct.unpack(order+'I', rawval)[0]
        return cls(tag, type_, count, val, isOffset)

    def toBytes(self, destBuf, newOffset=None, imgDataOffset=None, destOffset=0, order="="):
        if newOffset is None and self.isOffset:
            newOffset = self.val

        if self.isOffset:
            valFmt = 'L'
            val = newOffset
        elif self.isImageDataOffsetEntry() and imgDataOffset:
            # we are writing image data offsets within this entry; they are not themselves offset
            # for this to not be offset, there must be only one, since it's a long
            valFmt = 'L'
            val = imgDataOffset
        else:
            valFmt = lookupTagType(self.type).fmt * self.count
            val = self.val

        fmt = order+"HHI"+valFmt
        fmtSize = struct.calcsize(fmt)
        if fmtSize < 12:
            fmt += 'x'*(12-fmtSize)

        packing = [self.tag, self.type, self.count]
        if isinstance(val, tuple):
            packing += list(val)
        else:
            packing.append(val)

        struct.pack_into(fmt, destBuf, destOffset, *packing)
        return 12

    def asBytes(self, newOffset=None, imgDataOffset=None, order="="):
        buf = ctypes.create_string_buffer(self.byteSize())
        self.toBytes(buf, newOffset=newOffset, imgDataOffset=imgDataOffset, destOffset=0, order=order)
        return buf.raw

    def byteSize(self):
        return 12

    def getOffsetStartAndLength(self):
        if not self.isOffset:
            return None
        tagType = IFD_ENTRY_TYPECODE_TO_TAGTYPE[self.type].fmt
        l = struct.calcsize("=" + tagType * self.count)
        return self.val, l

    def getOffsetDataFormat(self):
        return IFD_ENTRY_TYPECODE_TO_TAGTYPE[self.type].fmt * self.count

    def isImageDataOffsetEntry(self):
        return self.tag in IMAGE_DATA_OFFSET_TAGS

    def isImageDataByteCountEntry(self):
        return self.tag in IMAGE_DATA_BYTECOUNT_TAGS

    def __str__(self):
        tagName = TAG_TO_NAME.get(self.tag, 'UNK')
        typeName = lookupTagType(self.type).type
        return "TiffIFDEntry(tag: %s (%d), type: %s (%d), count=%d, val=%s%s)" % \
               (tagName, self.tag, typeName, self.type, self.count, self.val,
                ' (offset)' if self.isOffset else '')


class TiffIFDEntryAndOffsetData(namedtuple("_TiffIFDEntryAndOffsetData", "entry offsetData")):
    """Simple pair structure to hold a TiffIFDEntry and its associated offset data, if any.

    If offset data is present (entry.isoffset is True), then it will be stored in offsetData as a tuple. If no
    offset data is present, then offsetData will be None.
    """
    def getData(self):
        """Returns tuple of data from offset if present, otherwise from entry
        """
        def tuplify(val):
            if isinstance(val, (tuple, list)):
                return tuple(val)
            return (val, )

        if self.entry.isOffset:
            return tuplify(self.offsetData)
        return tuplify(self.entry.val)


def lookupTagType(typecode):
    return IFD_ENTRY_TYPECODE_TO_TAGTYPE.get(typecode, UNKNOWN_TAGTYPE)


def calcReadsForOffsets(startLengthPairs, maxBuf=10**6, maxGap=1024):
    """Plans a sequence of file reads and seeks to cover all the spans of data in startLengthPairs.

    Parameters:
    -----------
    startLengthPairs: sequence of (int start, int length) pairs
        start is the offset position of an item of data, length is its size in bytes.

    maxBuf: positive integer, default 10^6 (1MB)
            Requests a largest size to use for file reads. Multiple contiguous image strips (or other data) of less
            than maxBuf in size will be read in a single read() call. If a single strip is larger than maxBuf, then
            it will still be read, in a single read call requesting exactly the strip size.

    maxGap: positive integer, default 1024 (1KB)
            Specifies the largest gap in meaningful data to tolerate within a single read() call. If two items of offset
            data for a single IFD are separated by more than maxGap of data not within the IFD, then they will be read
            in multiple read() calls. If they are separated by maxGap or less, then a single read() will be used and
            the irrelevant data in between simply ignored.

    Returns:
    --------
    sequence of (start, length) pairs, each representing a planned file read

    """
    # sort by starting position
    # we assume here that starts and offsets and generally sane - meaning (roughly) nonoverlapping
    if not startLengthPairs:
        return []
    startLengths = sorted(startLengthPairs, key=operator.itemgetter(0))

    bufStarts = []
    bufLens = []
    curStart, curLen = startLengths.pop(0)
    for start, length in startLengths:
        gap = start - (curStart + curLen)
        newLen = start + length - curStart
        if gap > maxGap or newLen > maxBuf:
            bufStarts.append(curStart)
            bufLens.append(curLen)
            curStart = start
            curLen = length
        else:
            curLen = newLen
    bufStarts.append(curStart)
    bufLens.append(curLen)
    return zip(bufStarts, bufLens)

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