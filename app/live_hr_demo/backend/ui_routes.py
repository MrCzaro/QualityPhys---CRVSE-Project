"""
FastHTML UI partial routes for the live HR demo.

These routes return server-rendered FastHTML components intended for HTMX or
small frontend swaps. They live beside the existing JSON API routes, which are
kept for smoke tests, diagnostics, and programmatic access.
"""
from __future__ import annotations
import json 
from fasthtml.common import *
from ui.result_components import (
    measurement_result_cards,
    measurement_result_cards_placeholder,
    model_prediction_summary_panel,
    model_prediction_summary_placeholder,
    repeatability_table,
    repeatability_table_placeholder,
    roi_analysis_summary_panel,
    roi_analysis_summary_placeholder,
    signal_summary_cards,
    signal_summary_cards_placeholder,
)

def _clean_query_value(value: str | None, fallback: str) -> str:
    """
    Normalize a query parameter for display.

    Parameters
    ----------
    value:
        Raw query parameter value.

    fallback:
        Text used when the value is empty or missing.

    Returns
    -------
    str
        Clean display value.
    """
    if value is None:
        return fallback

    value = str(value).strip()

    if value == "":
        return fallback

    return value


async def _read_json_object(request) -> dict:
    """
    Read a JSON request body as a dictionary.

    Returns an empty dictionary when the body is missing, malformed, or not a
    JSON object.
    """

    try:
        payload = await request.json()
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def _clean_payload_value(payload: dict, key: str, fallback: str) -> str:
    """
    Normalize a JSON payload field for display.
    """

    return _clean_query_value(payload.get(key), fallback)


def _demo_repeatability_runs() -> list[dict]:
    """
    Build demo rows for the repeatability table route.

    Returns
    -------
    list[dict]
        Example prediction runs formatted for UI rendering.
    """

    return [
        {
            "run_index": "1",
            "model_hr": "74.3 bpm",
            "spectral": "75.0 bpm",
            "difference": "-0.7 bpm",
            "green_sqi": "0.987 / good",
            "pos_sqi": "0.988 / good",
            "chrom_sqi": "0.987 / good",
            "used_seconds": "8.00",
            "samples": "240",
            "fps": "30.00",
        },
        {
            "run_index": "2",
            "model_hr": "74.6 bpm",
            "spectral": "75.0 bpm",
            "difference": "-0.4 bpm",
            "green_sqi": "0.986 / good",
            "pos_sqi": "0.989 / good",
            "chrom_sqi": "0.986 / good",
            "used_seconds": "8.00",
            "samples": "240",
            "fps": "30.00",
        },
    ]


def _parse_repeatability_runs_json(runs_json: str | None) -> list[dict]:
    """
    Parse repeatability table rows from a JSON query parameter.

    Parameters
    ----------
    runs_json:
        JSON-encoded list of repeatability run dictionaries.

    Returns
    -------
    list[dict]
        Parsed repeatability rows. Returns an empty list when the input is
        missing, invalid, or not a list.
    """

    if runs_json is None:
        return []

    try:
        parsed = json.loads(runs_json)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    rows = []

    for item in parsed:
        if isinstance(item, dict):
            rows.append(item)

    return rows


def _parse_repeatability_runs_payload(payload: dict) -> list[dict]:
    """
    Parse repeatability table rows from a JSON POST payload.
    """

    runs = payload.get("runs")

    if not isinstance(runs, list):
        return []

    rows = []

    for item in runs:
        if isinstance(item, dict):
            rows.append(item)

    return rows

def register_ui_routes(rt) -> None:
    """
    Register FastHTML UI partial routes.

    Parameters
    ----------
    rt:
        FastHTML route decorator returned by ``fast_app``.

    Returns
    -------
    None
    """

    @rt("/ui/measurement-results-placeholder")
    def measurement_results_placeholder() -> FT:
        """
        Render placeholder measurement result cards.

        Returns
        -------
        FT
            FastHTML component containing the default measurement result cards.
        """

        return measurement_result_cards_placeholder()

    @rt("/ui/measurement-results")
    def measurement_results(
        spectral_hr: str | None = None,
        spectral_detail: str | None = None,
        model_hr: str | None = None,
        model_detail: str | None = None,
        model_difference: str | None = None,
        model_difference_detail: str | None = None,
        quality: str | None = None,
        quality_detail: str | None = None,
    ) -> FT:
        """
        Render measurement result cards from query parameters.

        Parameters
        ----------
        spectral_hr:
            Display value for the primary spectral-consensus HR card.

        spectral_detail:
            Detail text for the primary spectral-consensus HR card.

        model_hr:
            Display value for the experimental model HR card.

        model_detail:
            Detail text for the experimental model HR card.

        model_difference:
            Display value for the model-vs-spectral agreement card.

        model_difference_detail:
            Detail text for the model-vs-spectral agreement card.

        quality:
            Display value for the measurement-quality card.

        quality_detail:
            Detail text for the measurement-quality card.

        Returns
        -------
        FT
            FastHTML component containing server-rendered measurement cards.
        """

        return measurement_result_cards(
            spectral_hr_value=_clean_query_value(
                spectral_hr,
                "Not analyzed yet",
            ),
            spectral_hr_detail=_clean_query_value(
                spectral_detail,
                "Primary estimate: spectral consensus",
            ),
            model_hr_value=_clean_query_value(
                model_hr,
                "Not predicted yet",
            ),
            model_hr_detail=_clean_query_value(
                model_detail,
                "Experimental CRVSE PhysFormer output",
            ),
            model_difference_value=_clean_query_value(
                model_difference,
                "Not predicted yet",
            ),
            model_difference_detail=_clean_query_value(
                model_difference_detail,
                "Agreement diagnostic",
            ),
            quality_value=_clean_query_value(
                quality,
                "Not analyzed yet",
            ),
            quality_detail=_clean_query_value(
                quality_detail,
                "Signal quality gate",
            ),
        )

    @rt("/ui/signal-summary-placeholder")
    def signal_summary_placeholder() -> FT:
        """
        Render placeholder GREEN / POS / CHROM diagnostic signal cards.

        Returns
        -------
        FT
            FastHTML component containing placeholder signal summary cards.
        """

        return signal_summary_cards_placeholder()

    @rt("/ui/signal-summary")
    def signal_summary(
        green: str | None = None,
        green_detail: str | None = None,
        pos: str | None = None,
        pos_detail: str | None = None,
        chrom: str | None = None,
        chrom_detail: str | None = None,
    ) -> FT:
        """
        Render GREEN / POS / CHROM diagnostic signal cards from query parameters.

        Parameters
        ----------
        green:
            Display value for the GREEN signal card.

        green_detail:
            Detail text for the GREEN signal card.

        pos:
            Display value for the POS signal card.

        pos_detail:
            Detail text for the POS signal card.

        chrom:
            Display value for the CHROM signal card.

        chrom_detail:
            Detail text for the CHROM signal card.

        Returns
        -------
        FT
            FastHTML component containing server-rendered diagnostic signal cards.
        """

        return signal_summary_cards(
            green_value=_clean_query_value(
                green,
                "Not analyzed yet",
            ),
            green_detail=_clean_query_value(
                green_detail,
                "Classical green-channel signal",
            ),
            pos_value=_clean_query_value(
                pos,
                "Not analyzed yet",
            ),
            pos_detail=_clean_query_value(
                pos_detail,
                "Plane-orthogonal-to-skin signal",
            ),
            chrom_value=_clean_query_value(
                chrom,
                "Not analyzed yet",
            ),
            chrom_detail=_clean_query_value(
                chrom_detail,
                "Chrominance-based signal",
            ),
        )

    @rt("/ui/repeatability-table-placeholder")
    def repeatability_table_placeholder_route() -> FT:
        """
        Render an empty live prediction repeatability table.

        Returns
        -------
        FT
            FastHTML component containing the repeatability table shell.
        """

        return repeatability_table_placeholder()

    @rt("/ui/repeatability-table-demo")
    def repeatability_table_demo() -> FT:
        """
        Render a demo live prediction repeatability table.

        Returns
        -------
        FT
            FastHTML component containing example repeatability rows.
        """

        return repeatability_table(
            runs=_demo_repeatability_runs(),
        )
    
    @rt("/ui/repeatability-table")
    def repeatability_table_route(
        runs_json: str | None = None,
    ) -> FT:
        """
        Render the live prediction repeatability table from JSON query data.

        Parameters
        ----------
        runs_json:
            JSON-encoded list of repeatability run dictionaries.

        Returns
        -------
        FT
            FastHTML component containing the repeatability table.
        """

        return repeatability_table(
            runs=_parse_repeatability_runs_json(runs_json),
        )
    
    @rt("/ui/repeatability-table-json")
    async def repeatability_table_json(request) -> FT:
        """
        Render the live prediction repeatability table from a JSON POST body.
        """

        payload = await _read_json_object(request)

        return repeatability_table(
            runs=_parse_repeatability_runs_payload(payload),
        )
    
    @rt("/ui/roi-analysis-summary-placeholder")
    def roi_analysis_summary_placeholder_route() -> FT:
        """
        Render the placeholder ROI analysis summary panel.

        Returns
        -------
        FT
            FastHTML component containing the placeholder ROI analysis summary.
        """

        return roi_analysis_summary_placeholder()


    @rt("/ui/roi-analysis-summary")
    def roi_analysis_summary_route(
        status: str | None = None,
        sample_count: str | None = None,
        duration_s: str | None = None,
        estimated_fps: str | None = None,
        spectral_consensus: str | None = None,
        green_summary: str | None = None,
        pos_summary: str | None = None,
        chrom_summary: str | None = None,
        raw_response: str | None = None,
    ) -> FT:
        """
        Render the ROI analysis summary panel from query parameters.

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
            Full backend response text.

        Returns
        -------
        FT
            FastHTML component containing the ROI analysis summary.
        """

        return roi_analysis_summary_panel(
            status=_clean_query_value(status, "Not analyzed yet"),
            sample_count=_clean_query_value(sample_count, "none"),
            duration_s=_clean_query_value(duration_s, "none"),
            estimated_fps=_clean_query_value(estimated_fps, "none"),
            spectral_consensus=_clean_query_value(spectral_consensus, "none"),
            green_summary=_clean_query_value(
                green_summary,
                "No GREEN signal analysis yet.",
            ),
            pos_summary=_clean_query_value(
                pos_summary,
                "No POS signal analysis yet.",
            ),
            chrom_summary=_clean_query_value(
                chrom_summary,
                "No CHROM signal analysis yet.",
            ),
            raw_response=_clean_query_value(
                raw_response,
                "No ROI series analyzed yet.",
            ),
        )

    @rt("/ui/roi-analysis-summary-json")
    async def roi_analysis_summary_json(request) -> FT:
        """
        Render the ROI analysis summary panel from a JSON POST body.
        """

        payload = await _read_json_object(request)

        return roi_analysis_summary_panel(
            status=_clean_payload_value(payload, "status", "Not analyzed yet"),
            sample_count=_clean_payload_value(payload, "sample_count", "none"),
            duration_s=_clean_payload_value(payload, "duration_s", "none"),
            estimated_fps=_clean_payload_value(payload, "estimated_fps", "none"),
            spectral_consensus=_clean_payload_value(
                payload,
                "spectral_consensus",
                "none",
            ),
            green_summary=_clean_payload_value(
                payload,
                "green_summary",
                "No GREEN signal analysis yet.",
            ),
            pos_summary=_clean_payload_value(
                payload,
                "pos_summary",
                "No POS signal analysis yet.",
            ),
            chrom_summary=_clean_payload_value(
                payload,
                "chrom_summary",
                "No CHROM signal analysis yet.",
            ),
            raw_response=_clean_payload_value(
                payload,
                "raw_response",
                "No ROI series analyzed yet.",
            ),
        )


    @rt("/ui/model-prediction-summary-placeholder")
    def model_prediction_summary_placeholder_route() -> FT:
        """
        Render the placeholder model prediction summary panel.

        Returns
        -------
        FT
            FastHTML component containing the placeholder model prediction summary.
        """

        return model_prediction_summary_placeholder()


    @rt("/ui/model-prediction-summary")
    def model_prediction_summary_route(
        status: str | None = None,
        model_hr: str | None = None,
        spectral_consensus: str | None = None,
        model_difference: str | None = None,
        green_summary: str | None = None,
        pos_summary: str | None = None,
        chrom_summary: str | None = None,
        original_duration_s: str | None = None,
        used_duration_s: str | None = None,
        used_samples: str | None = None,
        source_fps: str | None = None,
        raw_response: str | None = None,
    ) -> FT:
        """
        Render the model prediction summary panel from query parameters.

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
            Estimated source FPS.

        raw_response:
            Compact backend prediction response.

        Returns
        -------
        FT
            FastHTML component containing the model prediction summary.
        """

        return model_prediction_summary_panel(
            status=_clean_query_value(status, "Not predicted yet"),
            model_hr=_clean_query_value(model_hr, "none"),
            spectral_consensus=_clean_query_value(spectral_consensus, "none"),
            model_difference=_clean_query_value(model_difference, "none"),
            green_summary=_clean_query_value(
                green_summary,
                "No GREEN model-side spectral summary yet.",
            ),
            pos_summary=_clean_query_value(
                pos_summary,
                "No POS model-side spectral summary yet.",
            ),
            chrom_summary=_clean_query_value(
                chrom_summary,
                "No CHROM model-side spectral summary yet.",
            ),
            original_duration_s=_clean_query_value(original_duration_s, "none"),
            used_duration_s=_clean_query_value(used_duration_s, "none"),
            used_samples=_clean_query_value(used_samples, "none"),
            source_fps=_clean_query_value(source_fps, "none"),
            raw_response=_clean_query_value(
                raw_response,
                "No live model prediction run yet.",
            ),
        )

    @rt("/ui/model-prediction-summary-json")
    async def model_prediction_summary_json(request) -> FT:
        """
        Render the model prediction summary panel from a JSON POST body.
        """

        payload = await _read_json_object(request)

        return model_prediction_summary_panel(
            status=_clean_payload_value(payload, "status", "Not predicted yet"),
            model_hr=_clean_payload_value(payload, "model_hr", "none"),
            spectral_consensus=_clean_payload_value(
                payload,
                "spectral_consensus",
                "none",
            ),
            model_difference=_clean_payload_value(
                payload,
                "model_difference",
                "none",
            ),
            green_summary=_clean_payload_value(
                payload,
                "green_summary",
                "No GREEN model-side spectral summary yet.",
            ),
            pos_summary=_clean_payload_value(
                payload,
                "pos_summary",
                "No POS model-side spectral summary yet.",
            ),
            chrom_summary=_clean_payload_value(
                payload,
                "chrom_summary",
                "No CHROM model-side spectral summary yet.",
            ),
            original_duration_s=_clean_payload_value(
                payload,
                "original_duration_s",
                "none",
            ),
            used_duration_s=_clean_payload_value(
                payload,
                "used_duration_s",
                "none",
            ),
            used_samples=_clean_payload_value(payload, "used_samples", "none"),
            source_fps=_clean_payload_value(payload, "source_fps", "none"),
            raw_response=_clean_payload_value(
                payload,
                "raw_response",
                "No live model prediction run yet.",
            ),
        )

