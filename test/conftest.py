import pytest
import thunder

@pytest.fixture(scope='module', params=['spark'])
def context(request):
    if request.param == 'local':
        thunder.setup()
    if request.param == 'spark':
        thunder.setup(spark=True)

