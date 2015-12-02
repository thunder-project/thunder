def smallfloat(dtype):
    """
    Returns the smallest floating point dtype to which the passed dtype can be safely cast.

    For integers and unsigned ints, this will generally be next floating point type larger than the integer type. So
    for instance, smallest_float_type('uint8') -> dtype('float16'), smallest_float_type('int16') -> dtype('float32'),
    smallest_float_type('uint32') -> dtype('float64').

    This function relies on numpy's promote_types function.
    """
    from numpy import dtype as dtypefunc
    from numpy import promote_types
    intype = dtypefunc(dtype)
    compsize = max(2, intype.itemsize)  # smallest float is at least 16 bits
    comptype = dtypefunc('=f'+str(compsize))  # compare to a float of the same size
    return promote_types(intype, comptype)

def check_path(path, credentials=None):
    """
    Check that specified output path does not already exist

    The ValueError message will suggest calling with overwrite=True; this function is expected to be
    called from the various output methods that accept an 'overwrite' keyword argument.
    """
    from thunder.data.fileio.readers import get_file_reader
    reader = get_file_reader(path)(credentials=credentials)
    existing = reader.list(path, directories=True)
    if existing:
        raise ValueError("Path %s appears to already exist. Specify a new directory, "
                         "or call " % path + "with overwrite=True to overwrite.")

def connection_with_anon(credentials, anon=True):
    """
    Connect to S3 with automatic handling for anonymous access.

    Parameters
    ----------
    credentials : dict
        AWS access key ('access') and secret access key ('secret')

    anon : boolean, optional, default = True
        Whether to make an anonymous connection if credentials fail to authenticate
    """
    from boto.s3.connection import S3Connection
    from boto.exception import NoAuthHandlerFound

    try:
        conn = S3Connection(aws_access_key_id=credentials['access'],
                            aws_secret_access_key=credentials['secret'])
        return conn

    except NoAuthHandlerFound:
        if anon:
            conn = S3Connection(anon=True)
            return conn
        else:
            raise