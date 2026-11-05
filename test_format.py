import sys


class Example:
    def __init__(self, bar):
        if bar:
            bar += 1
            bar = bar * bar
        else:
            some_string = "Hello World"
            print(some_string, sys.path)
