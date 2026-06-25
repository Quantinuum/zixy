..
   Custom class template to make sphinx-autosummary list the full API doc after
   the summary. See https://github.com/sphinx-doc/sphinx/issues/7912

{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :exclude-members: __module__, __weakref__, __dict__, __annotations__, __dataclass_params__, __dataclass_fields__, __match_args__, __orig_bases__, __parameters__, __firstlineno__, __abstractmethods__, __annotate_func__, __annotations_cache__, __static_attributes__, __protocol_attrs__, __subclasshook__, __class_getitem__, __init_subclass__, __slots__, index, slice, raise_spec_type_error
   :special-members: __init__, __repr__, __eq__, __len__, __iter__, __getitem__, __setitem__, __add__, __sub__, __mul__, __rmul__, __iadd__, __isub__, __imul__, __itruediv__, __truediv__, __rtruediv__, __pos__, __neg__, __abs__, __int__, __float__, __complex__
   :inherited-members:
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   {% endif %}
   {% endblock %}
