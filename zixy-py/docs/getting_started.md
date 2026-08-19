---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_mode: force
---

# Getting Started

## Installation

### From PyPI

zixy is available [via PyPI](https://pypi.org/project/zixy/) and can be installed using your preferred
package manager, such as `pip`.

```bash
pip install zixy
```

zixy supports Python 3.11 and later.

The source code is available in the public [GitHub repository](https://github.com/quantinuum/zixy).

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
available in the `zixy.qubit.pauli` subpackage, with the qubits forming a basis for their definition
available in `zixy.qubit`.

```{code-cell} ipython3
import zixy.qubit as zq
import zixy.qubit.pauli as zqp

qubits = zq.Qubits.from_count(4)
print(len(qubits))

strings = zqp.Strings.from_str("X0 Y1 Z3", qubits)
print(str(strings))

terms = zqp.RealTerms.from_str("X0 Y1 Z3")
print(str(terms))
```

## Next steps

- Work through the [example notebooks](examples/basics.ipynb) for a tour of the rest of the library,
  including electronic Hamiltonians and fermionic mappings.
- Browse the API reference for details on the container types (`Coeff`, `Cmpnt`, `Term`) that
  the rest of the library builds on.
