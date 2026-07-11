"""
Reusable UI result components for the live HR demo.

These components are shared by the main live demo layout and, later, by
FastHTML/HTMX partial routes that render measurement results from backend
analysis and model prediction outputs.
"""
from __future__ import annotations
from fasthtml.common import *
from monsterui.all import *


def repeatability_table_placeholder() -> FT:
    """
    Render an empty live prediction repeatability table.

    Returns
    -------
    FT
        FastHTML component containing the repeatability table shell.
    """
    return repeatability_table(
        runs=[],
    )


def repeatability_table(runs: list[dict]) -> FT:
    """
    Render the live model prediction repeatability table.

    Parameters
    ----------
    runs:
        List of prediction run dictionaries. Each run may contain model HR,
        spectral consensus, model difference, SQI values, used duration,
        sample count, and source FPS.

    Returns
    -------
    FT
        FastHTML component containing the repeatability table.
    """
    if len(runs) == 0:
        table_body = Tbody(
            Tr(
                Td("No live model prediction runs yet.", colspan="10",cls="py-3 text-slate-500"),
                cls="border-b border-slate-100",), id="live-model-repeatability-table-body")
    else:
        table_body = Tbody(
            *[
                Tr(
                    Td(str(run.get("run_index", "")), cls="py-2 pr-3"),
                    Td(
                        str(run.get("model_hr", "none")),
                        cls="py-2 pr-3 font-medium",
                    ),
                    Td(str(run.get("spectral", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("difference", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("green_sqi", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("pos_sqi", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("chrom_sqi", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("used_seconds", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("samples", "none")), cls="py-2 pr-3"),
                    Td(str(run.get("fps", "none")), cls="py-2 pr-3"),
                    cls="border-b border-slate-100 text-slate-800",
                )
                for run in runs
            ],
            id="live-model-repeatability-table-body",
        )

    return Div(
        Div(
            Div(
                "Live prediction repeatability table",
                cls="text-sm font-semibold text-slate-900",
            ),
            Div(
                "Each row is one model-prediction run using the current ROI "
                "sample buffer. Starting a new main measurement clears this table.",
                cls="text-xs text-slate-600",
            ),
            cls="mb-2",
        ),
        Div(
            Table(
                Thead(
                    Tr(
                        Th("Run", cls="py-2 pr-3"),
                        Th("Model HR", cls="py-2 pr-3"),
                        Th("Spectral", cls="py-2 pr-3"),
                        Th("Diff", cls="py-2 pr-3"),
                        Th("GREEN SQI", cls="py-2 pr-3"),
                        Th("POS SQI", cls="py-2 pr-3"),
                        Th("CHROM SQI", cls="py-2 pr-3"),
                        Th("Used s", cls="py-2 pr-3"),
                        Th("Samples", cls="py-2 pr-3"),
                        Th("FPS", cls="py-2 pr-3"),
                        cls="border-b border-slate-200 text-left text-slate-500",
                    )
                ),
                table_body,
                cls="w-full border-collapse text-xs",
            ),
            cls="overflow-x-auto",
        ),
        id="live-model-repeatability-container",
        cls="mt-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
    )


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
    value_cls = "mt-2 break-normal text-2xl font-bold leading-tight text-slate-900 sm:text-3xl"

    if variant == "model":
        card_cls = "min-w-0 border-rose-100 shadow-sm"
        detail_cls = "mt-2 break-words text-xs leading-snug text-rose-700"
        value_cls = "mt-2 break-normal text-2xl font-bold leading-tight text-slate-900"

    label_cls = "text-[11px] font-semibold uppercase leading-tight tracking-wide text-slate-500"

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

def measurement_result_cards(
    spectral_hr_value: str,
    spectral_hr_detail: str,
    model_hr_value: str,
    model_hr_detail: str,
    model_difference_value: str,
    model_difference_detail: str,
    quality_value: str,
    quality_detail: str,
) -> FT:
    """
    Render the main four measurement result cards.

    Parameters
    ----------
    spectral_hr_value:
        Main text for the primary spectral-consensus HR card.

    spectral_hr_detail:
        Detail text for the primary spectral-consensus HR card.

    model_hr_value:
        Main text for the experimental model HR card.

    model_hr_detail:
        Detail text for the experimental model HR card.

    model_difference_value:
        Main text for the model-vs-spectral agreement card.

    model_difference_detail:
        Detail text for the model-vs-spectral agreement card.

    quality_value:
        Main text for the measurement-quality card.

    quality_detail:
        Detail text for the measurement-quality card.

    Returns
    -------
    FT
        FastHTML component containing the four main result cards.
    """
    return Div(
        metric_result_card(
            label="Estimated HR",
            value=spectral_hr_value,
            detail=spectral_hr_detail,
            value_id="spectral-consensus-summary",
        ),
        metric_result_card(
            label="Model Estimated HR",
            value=model_hr_value,
            detail=model_hr_detail,
            value_id="live-model-hr-summary",
            variant="model",
        ),
        metric_result_card(
            label="Model - spectral",
            value=model_difference_value,
            detail=model_difference_detail,
            value_id="model-spectral-difference-summary",
        ),
        metric_result_card(
            label="Measurement Quality",
            value=quality_value,
            detail=quality_detail,
            value_id="measurement-quality-summary",
            detail_id="measurement-quality-detail",
        ),
        cls="mt-4 grid gap-3 sm:grid-cols-2",
    )



def measurement_result_cards_placeholder() -> FT:
    """
    Render default placeholder measurement result cards.

    Returns
    -------
    FT
        FastHTML component containing the default measurement result cards.
    """
    return measurement_result_cards(
        spectral_hr_value="Not analyzed yet",
        spectral_hr_detail="Primary estimate: spectral consensus",
        model_hr_value="Not predicted yet",
        model_hr_detail="Experimental CRVSE PhysFormer output",
        model_difference_value="Not predicted yet",
        model_difference_detail="Agreement diagnostic",
        quality_value="Not analyzed yet",
        quality_detail="Signal quality gate",
    )


def signal_summary_cards(
    green_value: str,
    green_detail: str,
    pos_value: str,
    pos_detail: str,
    chrom_value: str,
    chrom_detail: str,
) -> FT:
    """
    Render GREEN / POS / CHROM diagnostic signal summary cards.

    Parameters
    ----------
    green_value:
        Main text for the GREEN signal card.

    green_detail:
        Detail text for the GREEN signal card.

    pos_value:
        Main text for the POS signal card.

    pos_detail:
        Detail text for the POS signal card.

    chrom_value:
        Main text for the CHROM signal card.

    chrom_detail:
        Detail text for the CHROM signal card.

    Returns
    -------
    FT
        FastHTML component containing the three diagnostic signal cards.
    """
    return Div(
        metric_result_card(
            label="GREEN",
            value=green_value,
            detail=green_detail,
            value_id="green-signal-summary",
        ),
        metric_result_card(
            label="POS",
            value=pos_value,
            detail=pos_detail,
            value_id="pos-signal-summary",
        ),
        metric_result_card(
            label="CHROM",
            value=chrom_value,
            detail=chrom_detail,
            value_id="chrom-signal-summary",
        ),
        cls="grid gap-3 md:grid-cols-3",
    )


def signal_summary_cards_placeholder() -> FT:
    """
    Render default placeholder diagnostic signal summary cards.

    Returns
    -------
    FT
        FastHTML component containing placeholder GREEN / POS / CHROM cards.
    """
    return signal_summary_cards(
        green_value="Not analyzed yet",
        green_detail="Classical green-channel signal",
        pos_value="Not analyzed yet",
        pos_detail="Plane-orthogonal-to-skin signal",
        chrom_value="Not analyzed yet",
        chrom_detail="Chrominance-based signal",
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

    disabled_cls = "disabled:cursor-not-allowed disabled:opacity-45 disabled:shadow-none"
    button_cls = f"{variant_classes.get(variant, variant_classes['secondary'])} {disabled_cls}"

    return Button(label, id=element_id, cls=button_cls)

def roi_analysis_summary_panel(
    status: str,
    sample_count: str,
    duration_s: str,
    estimated_fps: str,
    spectral_consensus: str,
    green_summary: str,
    pos_summary: str,
    chrom_summary: str,
    raw_response: str,
) -> FT:
    """
    Render the backend ROI analysis summary panel.

    Parameters
    ----------
    status:
        Backend analysis status.

    sample_count:
        Number of ROI samples analyzed.

    duration_s:
        Signal duration in seconds.

    estimated_fps:
        Estimated ROI sampling frequency.

    spectral_consensus:
        Spectral consensus HR summary.

    green_summary:
        GREEN channel spectral summary.

    pos_summary:
        POS channel spectral summary.

    chrom_summary:
        CHROM channel spectral summary.

    raw_response:
        Full backend response text for diagnostics.

    Returns
    -------
    FT
        FastHTML component containing the ROI analysis summary.
    """

    return Div(
        Div(
            Div(
                "Backend rPPG signal analysis",
                cls="text-sm font-semibold text-slate-900",
            ),
            Div(
                "Server-rendered summary of the analyzed ROI time series.",
                cls="text-xs text-slate-600",
            ),
            cls="mb-3",
        ),
        Div(
            Div(
                Div("Status", cls="text-xs font-medium uppercase text-slate-500"),
                Div(status, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Samples", cls="text-xs font-medium uppercase text-slate-500"),
                Div(sample_count, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Duration", cls="text-xs font-medium uppercase text-slate-500"),
                Div(duration_s, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Estimated FPS", cls="text-xs font-medium uppercase text-slate-500"),
                Div(estimated_fps, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            cls="grid gap-3 sm:grid-cols-2 lg:grid-cols-4",
        ),
        Div(
            Div(
                "Spectral consensus",
                cls="text-xs font-medium uppercase text-slate-500",
            ),
            Div(
                spectral_consensus,
                cls="mt-1 text-lg font-semibold text-slate-900",
            ),
            cls="mt-3 rounded-xl border border-emerald-100 bg-emerald-50 p-3",
        ),
        Div(
            Div(
                Div("GREEN", cls="text-xs font-medium uppercase text-slate-500"),
                Div(green_summary, cls="mt-1 text-sm text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("POS", cls="text-xs font-medium uppercase text-slate-500"),
                Div(pos_summary, cls="mt-1 text-sm text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("CHROM", cls="text-xs font-medium uppercase text-slate-500"),
                Div(chrom_summary, cls="mt-1 text-sm text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            cls="mt-3 grid gap-3 md:grid-cols-3",
        ),
        Details(
            Summary(
                "Full backend JSON response",
                cls="mt-3 cursor-pointer text-sm font-medium text-slate-700",
            ),
            Pre(
                raw_response,
                cls=(
                    "mt-2 max-h-96 overflow-x-auto overflow-y-auto rounded-xl "
                    "border border-slate-200 bg-white p-4 text-xs text-slate-800"
                ),
            ),
        ),
        id="roi-series-analysis-output",
        cls=(
            "rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm"
        ),
    )


def roi_analysis_summary_placeholder() -> FT:
    """
    Render the default placeholder ROI analysis summary panel.

    Returns
    -------
    FT
        FastHTML component containing the placeholder ROI analysis summary.
    """

    return roi_analysis_summary_panel(
        status="Not analyzed yet",
        sample_count="none",
        duration_s="none",
        estimated_fps="none",
        spectral_consensus="none",
        green_summary="No GREEN signal analysis yet.",
        pos_summary="No POS signal analysis yet.",
        chrom_summary="No CHROM signal analysis yet.",
        raw_response="No ROI series analyzed yet.",
    )


def model_prediction_summary_panel(
    status: str,
    model_hr: str,
    spectral_consensus: str,
    model_difference: str,
    green_summary: str,
    pos_summary: str,
    chrom_summary: str,
    original_duration_s: str,
    used_duration_s: str,
    used_samples: str,
    source_fps: str,
    raw_response: str,
) -> FT:
    """
    Render the experimental model prediction summary panel.

    Parameters
    ----------
    status:
        Backend prediction status.

    model_hr:
        Experimental model HR estimate.

    spectral_consensus:
        Classical spectral consensus HR estimate.

    model_difference:
        Difference between model HR and spectral consensus.

    green_summary:
        GREEN channel spectral summary.

    pos_summary:
        POS channel spectral summary.

    chrom_summary:
        CHROM channel spectral summary.

    original_duration_s:
        Original ROI buffer duration.

    used_duration_s:
        Duration used by the model input window.

    used_samples:
        Number of samples used by the model input window.

    source_fps:
        Estimated ROI sampling frequency.

    raw_response:
        Compact backend response text for diagnostics.

    Returns
    -------
    FT
        FastHTML component containing the model prediction summary.
    """
    return Div(
        Div(
            Div(
                "Experimental model prediction",
                cls="text-sm font-semibold text-slate-900",
            ),
            Div(
                "Server-rendered summary of the CRVSE PhysFormer live prediction.",
                cls="text-xs text-slate-600",
            ),
            cls="mb-3",
        ),
        Div(
            Div(
                Div("Status", cls="text-xs font-medium uppercase text-slate-500"),
                Div(status, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Model HR", cls="text-xs font-medium uppercase text-slate-500"),
                Div(model_hr, cls="mt-1 text-lg font-semibold text-rose-700"),
                cls="rounded-xl border border-rose-100 bg-rose-50 p-3",
            ),
            Div(
                Div("Spectral HR", cls="text-xs font-medium uppercase text-slate-500"),
                Div(spectral_consensus, cls="mt-1 text-lg font-semibold text-slate-900"),
                cls="rounded-xl border border-emerald-100 bg-emerald-50 p-3",
            ),
            Div(
                Div("Model - Spectral", cls="text-xs font-medium uppercase text-slate-500"),
                Div(model_difference, cls="mt-1 text-lg font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            cls="grid gap-3 sm:grid-cols-2 lg:grid-cols-4",
        ),
        Div(
            Div(
                Div("GREEN", cls="text-xs font-medium uppercase text-slate-500"),
                Div(green_summary, cls="mt-1 text-sm text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("POS", cls="text-xs font-medium uppercase text-slate-500"),
                Div(pos_summary, cls="mt-1 text-sm text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("CHROM", cls="text-xs font-medium uppercase text-slate-500"),
                Div(chrom_summary, cls="mt-1 text-sm text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            cls="mt-3 grid gap-3 md:grid-cols-3",
        ),
        Div(
            Div(
                Div("Original duration", cls="text-xs font-medium uppercase text-slate-500"),
                Div(original_duration_s, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Used duration", cls="text-xs font-medium uppercase text-slate-500"),
                Div(used_duration_s, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Used samples", cls="text-xs font-medium uppercase text-slate-500"),
                Div(used_samples, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            Div(
                Div("Source FPS", cls="text-xs font-medium uppercase text-slate-500"),
                Div(source_fps, cls="mt-1 text-sm font-semibold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-3",
            ),
            cls="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4",
        ),
        Details(
            Summary(
                "Compact backend prediction response",
                cls="mt-3 cursor-pointer text-sm font-medium text-slate-700",
            ),
            Pre(
                raw_response,
                cls=(
                    "mt-2 max-h-96 overflow-x-auto overflow-y-auto rounded-xl "
                    "border border-slate-200 bg-white p-4 text-xs text-slate-800"
                ),
            ),
        ),
        id="live-model-prediction-output",
        cls="rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm",
    )


def model_prediction_summary_placeholder() -> FT:
    """
    Render the default placeholder model prediction summary panel.

    Returns
    -------
    FT
        FastHTML component containing the placeholder model prediction summary.
    """
    return model_prediction_summary_panel(
        status="Not predicted yet",
        model_hr="none",
        spectral_consensus="none",
        model_difference="none",
        green_summary="No GREEN model-side spectral summary yet.",
        pos_summary="No POS model-side spectral summary yet.",
        chrom_summary="No CHROM model-side spectral summary yet.",
        original_duration_s="none",
        used_duration_s="none",
        used_samples="none",
        source_fps="none",
        raw_response="No live model prediction run yet.",
    )