"""
Reusable UI result components for the live HR demo.

These components are shared by the main live demo layout and, later, by
FastHTML/HTMX partial routes that render measurement results from backend
analysis and model prediction outputs.
"""
from __future__ import annotations
from fasthtml.common import *
from monsterui.all import *


def metric_result_card(label: str, value: str, detail: str, value_id: str, detail_id: str | None = None, 
                       variant: str = "default") -> FT:
    """
    Render one compact measurement result card.

    Parameters
    ----------
    label:
        Small uppercase card label.

    value:
        Initial value shown in the main card area.

    detail:
        Initial explanatory text below the value.

    value_id:
        Element ID used by JavaScript to update the main value.

    detail_id:
        Optional element ID used by JavaScript to update the detail text.

    variant:
        Visual variant. Supported values: ``default`` and ``model``.

    Returns
    -------
    FT
        FastHTML component for one result card.
    """
    card_variant = CardT.default
    card_cls = "min-w-0 shadow-sm"
    detail_cls = "mt-2 break-words text-xs leading-snug text-slate-500"

    if variant == "model":
        card_cls = "min-w-0 border-rose-100 shadow-sm"
        detail_cls = "mt-2 break-words text-xs leading-snug text-rose-700"

    label_cls = ("text-[11px] font-semibold uppercase leading-tight tracking-wide text-slate-500")
    value_cls = ("mt-2 break-words text-2xl font-bold leading-tight text-slate-900 sm:text-3xl")

    if detail_id is None:
        detail_node = Div(detail, cls=detail_cls)
    else:
        detail_node = Div(detail, id=detail_id, cls=detail_cls)

    return Card(
        CardBody(
            Div(label, cls=label_cls),
            Div(value, id=value_id, cls=value_cls),
            detail_node,
            cls="p-4",
        ),
        cls=(card_variant, card_cls),
    )


def diagnostic_card(title: str, description: str | None = None, *children) -> FT:
    """
    Render one diagnostics section card.

    Parameters
    ----------
    title:
        Card heading.

    description:
        Optional short explanation below the heading.

    *children:
        FastHTML child components rendered inside the card body.

    Returns
    -------
    FT
        FastHTML diagnostics card component.
    """

    description_nodes = []

    if description is not None:
        description_nodes.append(
            P(
                description,
                cls="mb-3 text-sm leading-relaxed text-slate-600",
            )
        )

    return Card(
        CardBody(
            H3(title, cls="mb-2 text-lg font-semibold text-slate-900"),
            *description_nodes,
            *children,
            cls="p-5",
        ),
        cls="shadow-sm",
    )


def main_panel_card(title: str, description: str | None = None, *children) -> FT:
    """
    Render one main demo panel card.

    Parameters
    ----------
    title:
        Main panel heading.

    description:
        Optional short explanation below the heading.

    *children:
        FastHTML child components rendered inside the panel body.

    Returns
    -------
    FT
        FastHTML component for one main demo panel.
    """
    description_nodes = []
    if description is not None:
        description_nodes.append(P(description, cls="mb-3 text-sm leading-relaxed text-slate-600",))

    return Card(
        CardBody(
            H3(title, cls="mb-3 text-lg font-semibold text-slate-900"),
            *description_nodes,
            *children,
            cls="p-5",
        ),
        cls="shadow-sm",
    )


def demo_button(label: str, element_id: str, variant: str = "secondary") -> FT:
    """
    Render one live demo control button.

    Parameters
    ----------
    label:
        Button text.

    element_id:
        HTML element ID used by JavaScript event listeners.

    variant:
        Visual variant. Supported values are ``primary``, ``secondary``,
        ``measurement``, ``stop_measurement``, ``diagnostic``, ``analysis``,
        ``model``, and ``face``.

    Returns
    -------
    FT
        FastHTML button component.
    """
    variant_classes = {
        "primary": (
            "rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium "
            "text-white shadow-sm hover:bg-slate-700"
        ),
        "secondary": (
            "rounded-lg border border-slate-300 bg-white px-4 py-2 "
            "text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
        ),
        "measurement": (
            "rounded-lg bg-emerald-700 px-4 py-2 text-sm font-medium "
            "text-white shadow-sm hover:bg-emerald-600"
        ),
        "stop_measurement": (
            "rounded-lg bg-amber-600 px-4 py-2 text-sm font-medium "
            "text-white shadow-sm hover:bg-amber-500"
        ),
        "diagnostic": (
            "rounded-lg border border-slate-300 bg-white px-4 py-2 "
            "text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
        ),
        "analysis": (
            "rounded-lg bg-indigo-700 px-4 py-2 text-sm font-medium "
            "text-white shadow-sm hover:bg-indigo-600"
        ),
        "model": (
            "rounded-lg bg-rose-700 px-4 py-2 text-sm font-medium "
            "text-white shadow-sm hover:bg-rose-600"
        ),
        "face": (
            "rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium "
            "text-white shadow-sm hover:bg-blue-600"
        ),
    }

    button_cls = variant_classes.get(variant, variant_classes["secondary"])

    return Button(label, id=element_id, cls=button_cls)