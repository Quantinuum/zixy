# Getting Started

## Installation

### From PyPI

zixy is available [via PyPI](https://pypi.org/project/zixy/) and can be installed using your preferred
package manager, such as `pip`.

```bash
pip install zixy
```
zixy supports Python versions 3.11 through 3.13, on macOs, Linux, and Windows.

The source code is available on a public [GitHub](https://github.com/quantinuum/zixy) repository. If you have a feature request or think you have found a bug, feel free to raise a [GitHub issue](https://github.com/Quantinuum/zixy/issues).

### From source

Developers may wish to install from source. The recommended method is using `maturin` from the
top-level directory.

```bash
git clone https://github.com/quantinuum/zixy
cd zixy
maturin develop
```

## A minimal Pauli string example

zixy is written in Rust, with Python bindings enabled via [PyO3](https://github.com/pyo3/pyo3).
Pauli strings, one of the quantum algebraic objects built on top of Zixy's general containers, are
available in the `zixy.qubit.pauli` submodule, with the qubits forming a basis for their definition
available in `zixy.qubit`.

```python
import zixy.qubit as zq
import zixy.qubit.pauli as zqp

qubits = zq.Qubits.from_count(4)
print(len(qubits))  # 4

strings = zqp.Strings.from_str("X0 Y1 Z3", qubits)
print(str(strings))  # X0 Y1 Z3

terms = zqp.RealTerms.from_str("X0 Y1 Z3")
print(str(terms))  # (1, X0 Y1 Z3)
```

## Next steps

- Work through the [example notebooks](examples/basics.ipynb) for a tour of the rest of the library,
  including chemistry Hamiltonians and fermionic mappings.
- Browse the API reference below for details on the container types (`Coeff`, `Cmpnt`, `Term`) that
  the rest of the library builds on.
