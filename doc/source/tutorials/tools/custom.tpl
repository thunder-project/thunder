{% extends 'rst.tpl'%}

{% block markdowncell scoped %}
{{ cell.source | markdown2rst | replace('=','-')}}
{% endblock markdowncell %}