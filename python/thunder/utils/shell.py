import thunder
from thunder.utils.context import ThunderContext

try:
	from termcolor import colored
except ImportError:
	colored = lambda x, y: x

tsc = ThunderContext(sc)

print('')
print(colored('       IIIII            ', 'yellow'))
print(colored('       IIIII            ', 'yellow'))
print(colored('    IIIIIIIIIIIIIIIIIII ', 'yellow'))
print(colored('    IIIIIIIIIIIIIIIIII  ', 'yellow'))
print(colored('      IIIII             ', 'yellow'))
print(colored('     IIIII              ', 'yellow'))
print(colored('     IIIII              ', 'yellow') + 'Thunder')
print(colored('      IIIIIIIII         ', 'yellow') + 'version ' + thunder.__version__)
print(colored('       IIIIIII          ', 'yellow'))
print('')

print('A Thunder context is available as tsc')