import inspect

__all__ = ['DebuggerObject']

__author__ = 'Carlos Daniel Archundia Cejudo'

class DebuggerObject():
    def __init__(self, max_depth, indent_string='\t'):
        self.initial_depth = len(inspect.stack())
        self.max_depth = max_depth
        self.indent_string = indent_string
        self.depth = 0

    def current_depth(self):
        return len(inspect.stack())-1

    def depth_in_range(self, current_depth=None):
        if not current_depth:
            self.depth = self.current_depth() - self.initial_depth
        else:
            self.depth = current_depth - self.initial_depth
        if self.max_depth >= 0:
            return (self.depth <= self.max_depth)
        else:
            return True

    def print(self, *printlist, indent=True):
        if self.depth_in_range(self.current_depth()):
            indent = self.indent_string*(self.depth-1) if indent else ''
            for obj in printlist:
                if not isinstance(obj, str):
                    obj = str(obj)
                for line in obj.split('\n'):
                    print(indent+line)

if __name__ == '__main__':
    print('Debugger example.')
    Debugger = DebuggerObject(max_depth=3, indent_string='  ')

    def level_1():
        func_name = inspect.currentframe().f_code.co_name
        Debugger.print(func_name)

        level_2()
        level_2()

    def level_2():
        func_name = inspect.currentframe().f_code.co_name
        Debugger.print(func_name)

        level_3()
        level_3()
        level_3()

    def level_3():
        func_name = inspect.currentframe().f_code.co_name
        Debugger.print(func_name)

    level_1()