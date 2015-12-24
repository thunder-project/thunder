import pytest
import station

@pytest.fixture(scope='module', params=['local', 'spark'])
def eng(request):
    if request.param == 'local':
        return None
    if request.param == 'spark':
        station.start(spark=True)
        return station.engine()

@pytest.fixture(scope='module')
def engspark():
    station.start(spark=True)
    return station.engine()
