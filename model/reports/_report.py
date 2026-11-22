"""Shared HTML-report plumbing: Jinja2 rendering, Plotly embedding, tokens.

Kept separate from `_spherical_ops.py` on purpose: everything in that module
is `keras.ops` math, everything in this one is I/O and presentation (Python
strings, template rendering, JSON for the browser). Mixing them would blur
the "no python/numpy in the math path" boundary the scripts are built around.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render_report(template_name: str, context: Dict[str, Any], out_path: Path) -> Path:
    style = (TEMPLATES_DIR / "_style.css").read_text()
    template = _env.get_template(template_name)
    html = template.render(style=style, **context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def plotly_div(fig: Any, div_id: str, include_js: bool = False) -> str:
    """Renders a Plotly figure to an embeddable `<div>` + `<script>` snippet."""
    import plotly.io as pio

    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_js else False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def token_label(tokenizer: Any, token_id: int) -> str:
    """Human-readable token text: strips the BPE leading-space marker `Ġ`."""
    raw = tokenizer.id_to_token(int(token_id))
    if raw is None:
        return f"<id {token_id}>"
    text = raw.replace("Ġ", " ").replace("Ċ", "\\n")
    return text if text.strip() else repr(text)


def token_labels(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    return [token_label(tokenizer, tid) for tid in token_ids]
