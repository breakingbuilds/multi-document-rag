"""Rendering helpers for the Streamlit front-end (app.py).

    styles.py      the CSS injected once per page (citation chips, cards)
    components.py  functions that turn a RAGResponse / manifest / report
                   rows into Streamlit widgets

The CLI has its own renderer (src/console.py); both read the same objects.
"""
