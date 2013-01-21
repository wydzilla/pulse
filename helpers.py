from clint.textui import puts
import sys

def mputs(*s, **kwargs):
    if s is []:
        puts()
    for item in s:
        if not isinstance(item, str):
            item = str(item)
        puts(item, newline=False)
        puts(' ', newline=False)
    puts(**kwargs)

class _printer:
    def __init__(self):
        self.last_len = 0
        self.prev_print = False

    def __call__(self, s):
        if self.prev_print:
            sys.stdout.write('\b'*self.last_len)

        self.last_len = len(s)
        sys.stdout.write(s)
        self.prev_print = True
        sys.stdout.flush()

inprint = _printer()
