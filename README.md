# easygsvd

A pure-Python, friendly interface to the generalized singular value decomposition (GSVD).

`easygsvd` is built on NumPy/SciPy and provides:

- A simple function `gsvd(A, L)` for dense GSVD of a matrix pair $(A, L)$.
- A `GSVDResult` object exposing the core GSVD factors (`Uhat`, `Vhat`, `X`, `Y`, `c`, `s`, etc.).
- Convenience methods for:
  - orthogonal projectors onto `col(A)`, `ker(A)`, `col(L)`, `ker(L)`, etc.
  - oblique projectors associated with the pair `(A, L)`.
  - A-weighted and L-weighted oblique pseudoinverses.
- An **incremental API** for appending columns to `(A, L)` and updating the GSVD efficiently.

See [the documentation](https://github.com/jlindbloom/easygsvd/blob/main/easygsvd_docs.pdf) for more details.

> **Note:** This implementation assumes the stacked matrix `np.vstack([A, L])`
> has **full column rank**. If this condition is violated (initially or after appending columns), a `LinAlgError` is raised.

---

## Installation

```bash
pip install easygsvd
```


## Example usage

```python
import numpy as np
from easygsvd import gsvd

# Sizes of test matrices
M = 200
K = 195
N = 100

# Draw random test matrices with standard Gaussian entries
A = np.random.normal(size=(M, N))
L = np.random.normal(size=(K, N))

# Compute the economic GSVD of (A, L)
gsvd_result = gsvd(A, L, full_matrices=False)

# Compute the economic GSVD of (A, L)
gsvd_result_full = gsvd(A, L, full_matrices=True)
gsvd_full = gsvd(A, L, tol=1e-12, full_matrices=True)
