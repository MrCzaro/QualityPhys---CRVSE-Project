from fasthtml.common import FT, Script


def live_demo_script() -> FT:
    """
    Return the browser-side JavaScript include for the live HR demo.

    The implementation lives in ``static/live_demo.js`` so the FastHTML
    component stays small and the browser-only camera/canvas/orchestration
    code is kept outside the Python UI layer.

    Returns
    -------
    FT
        FastHTML script tag loading the live demo JavaScript file.
    """

    return Script(src="/static/live_demo.js")
