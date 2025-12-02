# easygsvd

A pure-Python, friendly interface to the generalized singular value decomposition (GSVD).

`easygsvd` is built on NumPy/SciPy and provides:

- A simple function `gsvd(A, L)` for dense GSVD of a matrix pair \((A, L)\).
- A `GSVDResult` object exposing the core GSVD factors (`Uhat`, `Vhat`, `X`, `Y`, `c`, `s`, etc.).
- Convenience methods for:
  - orthogonal projectors onto `col(A)`, `ker(A)`, `col(L)`, `ker(L)`, etc.
  - oblique projectors associated with the pair `(A, L)`.
  - A-weighted and L-weighted oblique pseudoinverses.
- An **incremental API** for appending columns to `(A, L)` and updating the GSVD efficiently.

> **Note:** This implementation assumes the stacked matrix  
> \[
> \begin{bmatrix} A \\ L \end{bmatrix}
> \]
> has **full column rank**. If this condition is violated (initially or after appending columns), a `LinAlgError` is raised.

---

## Installation

```bash
pip install easygsvd
