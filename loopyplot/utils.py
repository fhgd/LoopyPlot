import logging

def get_plain_logger(name):
    log = logging.getLogger(name)
    if log.hasHandlers():
        log.handlers.clear()
    log.addHandler(logging.NullHandler())
    return log


def enable_logger(log, level='info', format='long'):
    if format.lower() == 'long':
        format = '%(name)s: %(message)s'
    elif format.lower() == 'short':
        format = '%(message)s'
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(format))
    if log.hasHandlers():
        log.handlers.clear()
    log.addHandler(ch)

    ch.setLevel(logging.DEBUG)

    level = getattr(logging, level.upper())
    log.setLevel(level)
