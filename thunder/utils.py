import logging

def notsupported(mode):
    logging.getLogger('thunder').warn("Operation not supported in '%s' mode" % mode)
    pass

def check_spark():
    SparkContext = False
    try:
        from pyspark import SparkContext
    finally:
        return SparkContext

def check_options(option, valid):
    if option not in valid:
        raise ValueError("Option must be one of %s, got '%s'" % (str(valid)[1:-1], option))

def check_path(path, credentials=None):
    """
    Check that specified output path does not already exist

    The ValueError message will suggest calling with overwrite=True;
    this function is expected to be called from the various output methods
    that accept an 'overwrite' keyword argument.
    """
    from thunder.readers import get_file_reader
    reader = get_file_reader(path)(credentials=credentials)
    existing = reader.list(path, directories=True)
    if existing:
        raise ValueError('Path %s appears to already exist. Specify a new directory, '
                         'or call with overwrite=True to overwrite.' % path)

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

def connection_with_gs(name):
    """
    Connect to GS
    """
    import boto
    conn = boto.storage_uri(name, 'gs')
    return conn