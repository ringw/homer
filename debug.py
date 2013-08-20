DEBUG_MODULES = []

import inspect
def DEBUG():
  try:
    if len(DEBUG_MODULES) == 0: return False
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    if module.__name__ in DEBUG_MODULES:
      return True
  finally:
    return False
