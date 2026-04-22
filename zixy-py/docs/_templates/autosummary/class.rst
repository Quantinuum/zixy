..
   Custom class template to make sphinx-autosummary list the full API doc after
   the summary. See https://github.com/sphinx-doc/sphinx/issues/7912
{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :exclude-members: __module__, __weakref__, __dict__, __annotations__, __dataclass_params__, __dataclass_fields__, __match_args__, __orig_bases__, __parameters__, __firstlineno__, __abstractmethods__, __annotate_func__, __annotations_cache__, __static_attributes__, __protocol_attrs__, __subclasshook__    
   :special-members:
   :inherited-members:
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   {% endif %}
   {% endblock %}