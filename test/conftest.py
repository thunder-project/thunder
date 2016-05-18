import pytest
import station

def pytest_addoption(parser):
    parser.addoption("--engine", action="store", default="local", 
                     help="engine to run tests with")

@pytest.fixture(scope='module')
def eng(request):
    engine = request.config.getoption("--engine")
    if engine == 'local':
        return None
    if engine == 'spark':
        station.start(spark=True)
        print station
        return station.engine()
