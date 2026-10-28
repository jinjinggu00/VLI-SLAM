"""Python3.10 alias only; original sparse numerical code stays unchanged."""
import collections
import collections.abc
if not hasattr(collections, "Sequence"):
 collections.Sequence = collections.abc.Sequence

import numpy as np
# Original NumPy1.20 release table: these builtin aliases preserve astype/dtype behavior.
if "int" not in np.__dict__: np.int = int
if "float" not in np.__dict__: np.float = float
