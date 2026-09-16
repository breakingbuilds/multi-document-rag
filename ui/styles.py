"""
ui/styles.py
------------
The small amount of CSS the web app needs on top of Streamlit's defaults:
citation chips inside answers, the cited-source star, chunk cards and the
metric bars of the Evaluation block. Injected once per page by app.py.

Colours use Streamlit's own theme variables where they exist so the page
follows the light / dark theme the user picked.
"""

CSS = """
<style>
/* [n] citation chips inside the answer text */
.cite {
  display: inline-block; min-width: 1.4em; padding: 0 .35em; margin: 0 .05em;
  border-radius: .6em; font-size: .78em; font-weight: 600; line-height: 1.5em;
  text-align: center; vertical-align: text-bottom; cursor: help;
  background: #FCA311; color: #14213D;
}
/* answer body */
.answer { font-size: 1.05rem; line-height: 1.65; }
.answer.refusal { border-left: 4px solid #FCA311; padding-left: .7rem; }
/* one retrieved chunk */
.chunk {
  border: 1px solid rgba(128,128,128,.35); border-radius: .5rem; padding: .6rem .8rem;
  margin-bottom: .6rem; font-size: .9rem;
}
.chunk .head { font-weight: 600; margin-bottom: .25rem; }
.chunk .meta { opacity: .7; font-size: .8rem; margin-bottom: .35rem; font-family: monospace; }
.chunk pre {
  white-space: pre-wrap; word-break: break-word; margin: 0; font-size: .82rem;
  opacity: .9; max-height: 9rem; overflow: auto;
}
/* rewrite list */
.rewrite { font-family: monospace; font-size: .85rem; }
.rewrite .tag { display: inline-block; width: 6.5rem; opacity: .6; }
/* metric bars */
.metric { display: grid; grid-template-columns: 9rem 6.5rem 1fr; align-items: center; gap: .5rem; margin: .2rem 0; }
.metric .bar { height: .55rem; border-radius: .3rem; background: rgba(128,128,128,.25); overflow: hidden; }
.metric .bar > div { height: 100%; background: #10B981; }
.metric .na { opacity: .5; }
.small { opacity: .7; font-size: .85rem; }
.star { color: #FCA311; }
</style>
"""
