from subprocess import check_output


__version__ = '0.0.1'

try:
    hg_id = check_output(['hg', 'id', '-i']).decode().strip('\n')
    __version__ += '-{}'.format(hg_id.replace('+', '_modified'))
except FileNotFoundError:
    pass
