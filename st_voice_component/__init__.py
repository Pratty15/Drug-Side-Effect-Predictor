import os
import streamlit.components.v1 as components
from typing import Dict, Any

_RELEASE = True

if _RELEASE:
    # Point directly to the frontend directory which contains index.html
    _component_func = components.declare_component(
        "st_voice_component",
        path=os.path.join(os.path.dirname(__file__), "frontend"),
    )
else:
    _component_func = components.declare_component("st_voice_component", url="http://localhost:5173")

def voice_component(key: str = "voice_rec", **kwargs) -> Dict[str, Any]:
    val = _component_func(key=key, **kwargs)
    if val is None:
        return {"transcript": "", "is_final": False, "supported": True}
    if isinstance(val, dict):
        return val
    try:
        return dict(val)
    except Exception:
        return {"transcript": str(val), "is_final": False, "supported": True}
