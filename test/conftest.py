import pytest
import station

@pytest.fixture(scope='module', params=['local', 'spark'])
def eng(request):
    if request.param == 'local':
        return None
    if request.param == 'spark':
        station.setup(spark=True)
        return station.engine()

@pytest.fixture(scope='module')
def engspark():
    station.setup(spark=True)
    return station.engine()
