# Zixy

[![CI](https://github.com/quantinuum/zixy/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/quantinuum/zixy/actions?query=workflow%3A%22Unit+tests%22+branch%3Amain)
[![PyPI version](https://badge.fury.io/py/zixy.svg)](https://badge.fury.io/py/zixy)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](https://opensource.org/license/apache-2.0)

Zixy is a high performance library for the manipulation of Pauli strings and other quantum algebraic objects.

## Installation

### From PyPI

The package is available [via PyPI](https://pypi.org/project/zixy/) and can be installed using the preferred package manager,
such as `pip`.

```bash
pip install zixy
```

### From source

Developers may wish to install from source.
The recommended method is by using `maturin` from the top level directory.

```bash
git clone https://github.com/quantinuum/zixy
cd zixy
maturin develop
```

## Usage

### Overview

The package is written in Rust, with Python bindings enabled via [PyO3](https://github.com/pyo3/pyo3).
The base containers in the Python interface are defined in a general fashion.
The user-facing API separates these objects into three important classes:

- `Coeff`: a **coefficient**, defining a scalar value of varying types.
- `Cmpnt`: a **component**, defining a basic algebraic building block.
- `Term`: a **term**, consisting of a `Cmpnt` scaled by a `Coeff`.

Coefficients are separated into built-in Python types such as `float` or `complex`, symbolic `sympy` expressions,
and roots of unity such as `Sign` and `ComplexSign`,
which have custom implementations.

Additionally, each such class has associated containers for collections of that type.
Implementations of these base classes may narrow their implementation depending on the coefficient type.

### Example

These general building blocks are sufficient to define many different quantum algebraic objects.
An example of one such specification of the objects in Zixy is Pauli strings.
These are made available in the `zixy.qubit.pauli` submodule,
with the qubits forming a basis for their definition available in `zixy.qubit`.

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

For other specifications such as computational basis vectors,
further [example notebooks](https://github.com/quantinuum/zixy/tree/main/zixy-py/examples) are available.

### Rust

The trait `NumRepr` (meaning "number representation") is implemented by all types that can represent the numbers multiplying components.
In collections, these are stored contiguously in vector-like structures.
Values of `NumRepr` types are closed under multiplication, but do not implement addition.
Implementers of `NumRepr` are

- `Unity`: a unit type representing `+1` and only `+1`. The corresponding collection container `UnityVec` only stores a length, the number of unit values.
- `Sign`: the square roots of unity (`+1`, `-1`). The corresponding collection container, `SignVec` is implemented as a bitset.
- `ComplexSign`: the fourth roots of unity (`+1`, `+i`, `-1`, `-i`). The corresponding collection container, `ComplexSignVec` is implemented as a paired bitset.

The `FieldElem` subtrait of `NumRepr` is implemented by types whose values form a field with multiplication and addition.
Implementers of both `NumRepr` and `FieldElem` use `Vec<T>` for collection storage, and are

- `Real`: Floating point value `f64`
- `Complex`: Complex floating point value `Complex64` from `num_complex` crate.

The Python interface further supports symbolic types via SymPy,
which are not currently available within the Rust library.

On the Rust side, components are stored contiguously in `CmpntList` types.
These implement the business logic applicable to their elements.
There are three main kinds of container that encapsulate `CmpntList` with generic coefficient type `C`

- `Terms<C: NumRepr>`: stores a `C::Vector` of the same length as the `CmpntList`, offers mutable access to the components and coefficients.
- `TermSet<C: NumRepr>`: stores a `C::Vector` of the same length as the `CmpntList` and via an indirect hash map, provides constant complexity lookup and enforces uniqueness among components. Offers mutable access only to the coefficients.
- `TermSum<C: FieldElem>`: works like `Set<C>` but with linear combination semantics and is only defined for `FieldElem`-implementing coefficient types.
