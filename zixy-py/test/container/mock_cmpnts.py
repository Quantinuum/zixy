from __future__ import annotations

from collections.abc import Sequence

from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet, ImplArray

"""
Mock-up a viewable component list system with classes extending:
    - basic array storage implementation (ImplArray)
    - single cmpnt view / owning object (Cmpnt)
    - multiple cmpnt view / owning object (Cmpnts)
"""


class StringsImplArray(ImplArray):
    def __init__(self):
        self._list = []

    @classmethod
    def from_list(cls, source: list[str]) -> StringsImplArray:
        out = cls()
        out._list = source
        return out

    @classmethod
    def from_len(cls, n: int) -> StringsImplArray:
        out = cls()
        out.resize(n)
        return out

    def resize(self, n: int):
        if n > len(self):
            n -= len(self)
            self._list.extend([""] * n)
        else:
            self._list = self._list[:n]

    def __len__(self) -> int:
        return len(self._list)

    def cmpnts_clone(self, inds: Sequence[int] | None) -> StringsImplArray:
        out = StringsImplArray()
        inds = range(len(self)) if inds is None else inds
        out._list = [self._list[i] for i in inds]
        return out

    def cmpnts_eq(
        self,
        inds: Sequence[int] | None,
        other: StringsImplArray,
        other_inds: Sequence[int] | None,
    ) -> bool:
        inds = range(len(self)) if inds is None else inds
        other_inds = range(len(other)) if other_inds is None else other_inds
        if len(inds) != len(other_inds):
            return False
        for l, r in zip(inds, other_inds, strict=False):
            if self._list[l] != other._list[r]:
                return False
        return True

    def cmpnt_copy_internal(self, i_dst: int, i_src: int):
        self._list[i_dst] = self._list[i_src]

    def cmpnt_copy_external(self, i_dst: int, src: StringsImplArray, i_src: int):
        self._list[i_dst] = src._list[i_src]

    def __eq__(self, other: StringsImplArray) -> bool:
        return self._list == other._list

    def same_as(self, other: StringsImplArray) -> bool:
        return self._list is other._list

    def _refresh_map(self, map: dict[str, int]):
        map.clear()
        map.update({s: i for i, s in enumerate(self._list)})

    def mapped_insert(self, map: dict[str, int], other: StringsImplArray, index: int) -> int:
        out = self.mapped_lookup(map, other, index)
        if out is None:
            out = len(self)
            self._list.append(other._list[index])
            self._refresh_map(map)
        else:
            self._list[out] = other._list[index]
        return out

    def mapped_lookup(self, map: dict[str, int], other: StringsImplArray, index: int) -> int | None:
        return map.get(other._list[index])

    def mapped_remove(self, map: dict[str, int], other: StringsImplArray, index: int) -> int | None:
        out = self.mapped_lookup(map, other, index)
        if out is None:
            return None
        self._list[out] = self._list[-1]
        self._list = self._list[:-1]
        self._refresh_map(map)
        return out

    def mapped_equal(self, map: dict[str, int], other: StringsImplArray) -> bool:
        if len(self) != len(other):
            return False
        return all(self.mapped_lookup(map, s) is not None for s in other._list)


CmpntSpecT = str


class String(Cmpnt[StringsImplArray, CmpntSpecT]):
    impl_type = StringsImplArray

    def __init__(self, source: CmpntSpecT):
        impl = self.impl_type()
        impl.resize(1)
        super().__init__(impl)
        self.set(source)

    def __repr__(self) -> str:
        return str(self._impl._list[self.index])

    def copy(self) -> String:
        return String(self)

    def set(self, source: str | String | None):
        if source is None:
            source = ""
        if isinstance(source, str):
            self._impl._list[self.index] = source
        elif isinstance(source, String):
            self._impl._list[self.index] = source._impl._list[source.index]
        else:
            String.raise_spec_type_error(source)


class Strings(Cmpnts[StringsImplArray, CmpntSpecT]):
    cmpnt_type = String

    def __init__(self, n: int = 0):
        super().__init__(StringsImplArray.from_len(n))


class StringSet(CmpntSet[StringsImplArray, CmpntSpecT]):
    cmpnts_type = Strings
    map_type = dict

    def __init__(self):
        super().__init__(StringsImplArray())


Strings._set_type = StringSet
