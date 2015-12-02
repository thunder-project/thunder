from thunder.data.fileio.readers import LocalFileReader, LocalParallelReader


def make(tmpdir, files):
    tmpdir.mkdir('foo')
    tmpdir.mkdir('bar')
    tmpdir.mkdir('foo/bar')
    for f in files:
        tmpdir.join(f).write('hi')

def parse(files):
    return [f.split('/')[-1] for f in files]


def test_parallel_flat(tmpdir):
    filenames = ['b', 'a', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_local_flat(tmpdir):
    filenames = ['b', 'a', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_parallel_recursive_flat(tmpdir):
    filenames = ['b', 'a', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_local_recursive_flat(tmpdir):
    filenames = ['a', 'b', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_parallel_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['c']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_local_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['c']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_parallel_recursive_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['a', 'c', 'b', 'q']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_local_recursive_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['a', 'c', 'b', 'q']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected
