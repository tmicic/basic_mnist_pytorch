import time

def print_sl(*args, **kwargs):
    print("\033[F\033[K", end='')
    print(*args, **kwargs)


