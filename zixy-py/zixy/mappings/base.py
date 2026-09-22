# Copyright 2026 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base interface for mappers between concrete types."""

from abc import ABC, abstractmethod
from typing import Any


class Mapper(ABC):
    """Base interface for transformations between domains.

    Concrete mapper classes specify the accepted input and corresponding output types on their
    ``apply`` implementations.
    """

    @abstractmethod
    def apply(self, value: Any, /) -> Any:
        """Apply this mapper to ``value``."""
        raise NotImplementedError
