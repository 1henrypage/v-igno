from src.components.nf import RealNVPFlow

import inspect

source = inspect.getsource(RealNVPFlow.forward)
print(source)
