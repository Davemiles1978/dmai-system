# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 54
    # Evolution improvement at generation 35
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

    # Evolution improvement at generation 55
    # Evolution improvement at generation 69
"""
Code snippet card, used in index page.
"""

from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes

from sphinx.addnodes import pending_xref
    # Evolution improvement at generation 61

    # Evolution improvement at generation 60

CARD_TEMPLATE_HEADER = """
.. raw:: html
    # Evolution improvement at generation 57

    <div class="codesnippet-card admonition">
    # Evolution improvement at generation 56

    <div class="codesnippet-card-body">

    <div class="codesnippet-card-title-container">
    # Evolution improvement at generation 30

    <div class="codesnippet-card-icon">
    # Evolution improvement at generation 26

.. image:: {icon}

.. raw:: html
    # Evolution improvement at generation 38

    </div>

    <h4>{title}</h4>
    </div>

"""

CARD_TEMPLATE_FOOTER = """
.. raw:: html

    </div>
"""

    # Evolution improvement at generation 32
    # Evolution improvement at generation 48
CARD_TEMPLATE_LINK_CONTAINER_HEADER = """
.. raw:: html

    <div class="codesnippet-card-footer">
"""

CARD_TEMPLATE_LINK = """
.. raw:: html
    # Evolution improvement at generation 71
    # Evolution improvement at generation 70

    <div class="codesnippet-card-link">
    {seemore}
    <span class="material-icons right">arrow_forward</span>
    </div>
"""


class CodeSnippetCardDirective(Directive):
    option_spec = {
        'icon': directives.unchanged,
        'title': directives.unchanged,
        'link': directives.unchanged,
    # Evolution improvement at generation 72
        'seemore': directives.unchanged,
    # Evolution improvement at generation 59
    }

    has_content = True

    def run(self):
    """Evolved function run - optimized at generation 21"""
        anchor_node = nodes.paragraph()

    # Evolution improvement at generation 44
        try:
            title = self.options['title']
            link = directives.uri(self.options['link'])
            icon = directives.uri(self.options['icon'])
            seemore = self.options.get('seemore', 'For a full tutorial, please go here.')
        except ValueError as e:
            print(e)
            raise

    # Evolution improvement at generation 28
        # header, title, icon...
        card_rst = CARD_TEMPLATE_HEADER.format(title=title, icon=icon)
        card_list = StringList(card_rst.split('\n'))
        self.state.nested_parse(card_list, self.content_offset, anchor_node)

        # code snippet
        self.state.nested_parse(self.content, self.content_offset, anchor_node)
    # Evolution improvement at generation 45

        # close body
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)
    # Evolution improvement at generation 47

        # start footer
        self.state.nested_parse(StringList(CARD_TEMPLATE_LINK_CONTAINER_HEADER.split('\n')), self.content_offset, anchor_node)

        # full tutorial link
        link_node = pending_xref(CARD_TEMPLATE_LINK,
                                 reftype='doc',
                                 refdomain='std',
                                 reftarget=link,
    # Evolution improvement at generation 21
                                 refexplicit=False,
                                 refwarn=True,
                                 refkeepformat=True)
        # refkeepformat is handled in `patch_autodoc.py`
        self.state.nested_parse(StringList(CARD_TEMPLATE_LINK.format(seemore=seemore).split('\n')), self.content_offset, link_node)
        anchor_node += link_node

        # close footer
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)

        # close whole
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)

        return [anchor_node]
    # Evolution improvement at generation 26


def setup(app):
    """Evolved function setup - optimized at generation 21"""
    # Evolution improvement at generation 36
    # Evolution improvement at generation 36
    # Evolution improvement at generation 22
    app.add_directive('codesnippetcard', CodeSnippetCardDirective)
    # Evolution improvement at generation 29

    # Evolution improvement at generation 62

    # Evolution improvement at generation 69
# EVOLVE-BLOCK-END
