# https://johnpaton.net/posts/redirect-logging/
# This class allows you to redirect every print() to be logged instead.

import logging
import contextlib
import sys

class OutputLogger:
    def __init__(self, name="root", level="INFO", input='stdout'):
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)
        if input == 'stdout':
            self._redirector = contextlib.redirect_stdout(self)
        elif input == 'stderr':
            self._redirector = contextlib.redirect_stderr(self)
        else:
            raise ValueError(f'OutputLogger input should be stdout or stderr but got {input}')

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self): pass

    def __enter__(self):
        self._redirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # let contextlib do any exception handling here
        self._redirector.__exit__(exc_type, exc_value, traceback)
