  let cameraStream = null;
  let hasCapturedFrame = false;
  let roiSamplingScheduleId = null;
  let roiSamples = [];
  let roiSamplingStartMs = null;
  let roiSamplingInFlight = false;
  let roiSamplingRunId = 0;
  let mainMeasurementTimer = null;
  let mainMeasurementProgressTimer = null;
  let mainMeasurementInProgress = false;
  let measurementRevision = 0;
  let latestRoiAnalysisDisplayState = null;


    const MAIN_MEASUREMENT_DURATION_MS = 15000;

    const ROI_NAMES = ["forehead", "image_left_cheek", "image_right_cheek"];

    const ROI_COLORS = {
      forehead: "#7c3aed",
      image_left_cheek: "#16a34a",
      image_right_cheek: "#dc2626"
    };

    function buildServerPartialUrl(routePath, paramsObject = {}) {
      /*
      Build a FastHTML partial URL with non-null query parameters only.
      */

      const params = new URLSearchParams();

      for (const [key, value] of Object.entries(paramsObject)) {
        if (value !== null && value !== undefined) {
          params.set(key, String(value));
        }
      }

      const queryString = params.toString();

      if (queryString.length === 0) {
        return routePath;
      }

      return `${routePath}?${queryString}`;
    }

    async function fetchServerPartialHtml(routePath, paramsObject = {}, method = "GET") {
      /*
      Fetch a server-rendered FastHTML partial and return its HTML text.
      */

      const normalizedMethod = String(method).toUpperCase();
      let response = null;

      if (normalizedMethod === "POST") {
        response = await fetch(routePath, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(paramsObject ?? {})
        });
      } else {
        const url = buildServerPartialUrl(routePath, paramsObject);
        response = await fetch(url);
      }

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      return await response.text();
    }

    async function renderServerPartial({
      containerId,
      routePath,
      params = {},
      method = "GET",
      expectedRevision = null,
      missingMessage = "Server partial container not found.",
      errorMessage = "Could not render server partial:",
    } = {}) {
      /*
      Fetch a FastHTML partial and swap it into a known container.

      When expectedRevision is provided, the partial is only applied if the
      measurement revision is still current. This prevents delayed reset/result
      partials from overwriting newer UI state.
      */

      const hasRevisionGuard = Number.isFinite(expectedRevision);

      if (hasRevisionGuard && expectedRevision !== measurementRevision) {
        return false;
      }

      const container = document.getElementById(containerId);

      if (!container) {
        console.warn(missingMessage);
        return false;
      }

      try {
        const html = await fetchServerPartialHtml(routePath, params, method);

        if (hasRevisionGuard && expectedRevision !== measurementRevision) {
          return false;
        }

        container.innerHTML = html;
        updateFinalInterpretationFromCurrentDom();
        updateDemoReadinessPanel();
        return true;
      } catch (error) {
        console.error(errorMessage, error);
        return false;
      }
    }

    async function renderMeasurementResultCardsFromServer({
        spectralHr = null,
        spectralDetail = null,
        modelHr = null,
        modelDetail = null,
        modelDifference = null,
        modelDifferenceDetail = null,
        quality = null,
        qualityDetail = null,
    } = {}) {
        /*
        Fetch server-rendered measurement result cards and swap them into the page.
        */

        await renderServerPartial({
            containerId: "measurement-result-cards-container",
            routePath: "/ui/measurement-results",
            params: {
                spectral_hr: spectralHr,
                spectral_detail: spectralDetail,
                model_hr: modelHr,
                model_detail: modelDetail,
                model_difference: modelDifference,
                model_difference_detail: modelDifferenceDetail,
                quality: quality,
                quality_detail: qualityDetail,
            },
            missingMessage: "Measurement result cards container not found.",
            errorMessage: "Could not render measurement result cards:",
        });
    }


    async function renderMeasurementResultCardsPlaceholderFromServer({
      expectedRevision = null,
    } = {}) {
      /*
      Fetch the server-rendered placeholder for the main measurement cards.
      */

      await renderServerPartial({
        containerId: "measurement-result-cards-container",
        routePath: "/ui/measurement-results-placeholder",
        expectedRevision: expectedRevision,
        missingMessage: "Measurement result cards container not found.",
        errorMessage: "Could not render measurement result card placeholder:",
      });
    }



    async function renderSignalSummaryCardsFromServer({
      green = null,
      greenDetail = null,
      pos = null,
      posDetail = null,
      chrom = null,
      chromDetail = null,
    } = {}) {
      /*
      Fetch server-rendered GREEN / POS / CHROM diagnostic signal cards.
      */

      await renderServerPartial({
        containerId: "signal-summary-cards-container",
        routePath: "/ui/signal-summary",
        params: {
          green: green,
          green_detail: greenDetail,
          pos: pos,
          pos_detail: posDetail,
          chrom: chrom,
          chrom_detail: chromDetail,
        },
        missingMessage: "Signal summary cards container not found.",
        errorMessage: "Could not render signal summary cards:",
      });
    }


    async function renderSignalSummaryCardsPlaceholderFromServer({
      expectedRevision = null,
    } = {}) {
      /*
      Fetch the server-rendered placeholder for GREEN / POS / CHROM cards.
      */

      await renderServerPartial({
        containerId: "signal-summary-cards-container",
        routePath: "/ui/signal-summary-placeholder",
        expectedRevision: expectedRevision,
        missingMessage: "Signal summary cards container not found.",
        errorMessage: "Could not render signal summary card placeholder:",
      });
    }

    async function renderRoiAnalysisSummaryPlaceholderFromServer({
      expectedRevision = null,
    } = {}) {
      /*
      Fetch the server-rendered placeholder for the ROI analysis summary panel.
      */

      await renderServerPartial({
        containerId: "roi-series-analysis-output-container",
        routePath: "/ui/roi-analysis-summary-placeholder",
        expectedRevision: expectedRevision,
        missingMessage: "ROI analysis summary container not found.",
        errorMessage: "Could not render ROI analysis placeholder:",
      });
    }

    function buildCompactRoiAnalysisDebugResponse(data) {
      /*
      Build a compact diagnostic payload for the collapsible ROI analysis panel.

      The full backend response contains long signal arrays. Passing that whole
      object through a GET query parameter can exceed practical URL limits, so
      the server-rendered panel receives the key metadata and spectral summaries.
      */

      const compactSignals = {};

      for (const signalName of ["green", "pos", "chrom"]) {
        const signalData = data.signals?.[signalName];

        if (!signalData) {
          continue;
        }

        compactSignals[signalName] = {
          ok: signalData.ok,
          message: signalData.message,
          n_samples: signalData.n_samples,
          mean: signalData.mean,
          std: signalData.std,
          min: signalData.min,
          max: signalData.max,
          spectral: signalData.spectral ?? null,
        };
      }

      return {
        status: data.status,
        message: data.message,
        sample_count: data.sample_count,
        duration_s: data.duration_s,
        estimated_fps: data.estimated_fps,
        roi_names_used: data.roi_names_used ?? [],
        window_metadata: data.window_metadata ?? null,
        signals: compactSignals,
        notes: data.notes ?? [],
        omitted_from_compact_view: [
          "time_s",
          "signals.green.values",
          "signals.pos.values",
          "signals.chrom.values",
        ],
      };
    }

    async function renderRoiAnalysisSummaryFromServer({
      status = null,
      sampleCount = null,
      durationS = null,
      estimatedFps = null,
      spectralConsensus = null,
      greenSummary = null,
      posSummary = null,
      chromSummary = null,
      rawResponse = null,
    } = {}) {
      /*
      Fetch the server-rendered ROI analysis summary panel.
      */

      await renderServerPartial({
        containerId: "roi-series-analysis-output-container",
        routePath: "/ui/roi-analysis-summary-json",
        method: "POST",
        params: {
          status: status,
          sample_count: sampleCount,
          duration_s: durationS,
          estimated_fps: estimatedFps,
          spectral_consensus: spectralConsensus,
          green_summary: greenSummary,
          pos_summary: posSummary,
          chrom_summary: chromSummary,
          raw_response: rawResponse,
        },
        missingMessage: "ROI analysis summary container not found.",
        errorMessage: "Could not render ROI analysis summary:",
      });
    }

    async function renderModelPredictionSummaryPlaceholderFromServer({
      expectedRevision = null,
    } = {}) {
      /*
      Fetch the server-rendered placeholder for the model prediction summary panel.
      */

      await renderServerPartial({
        containerId: "live-model-prediction-output-container",
        routePath: "/ui/model-prediction-summary-placeholder",
        expectedRevision: expectedRevision,
        missingMessage: "Model prediction summary container not found.",
        errorMessage: "Could not render model prediction placeholder:",
      });
    }

    function buildCompactModelPredictionDebugResponse(data) {
      /*
      Build a compact diagnostic payload for the collapsible model prediction panel.
      */

      const modelInput = data.model_input ?? {};

      return {
        status: data.status,
        message: data.message ?? null,
        model_available: data.model_available ?? null,
        model_load: data.model_load ?? null,
        model_prediction: data.model_prediction ?? null,
        classical_analysis_status: data.classical_analysis_status ?? null,
        classical_analysis_message: data.classical_analysis_message ?? null,
        classical_spectral_summary: data.classical_spectral_summary ?? null,
        model_input: {
          input_shape: modelInput.input_shape ?? null,
          channel_order: modelInput.channel_order ?? null,
          target_length: modelInput.target_length ?? null,
          model_target_frames: modelInput.model_target_frames ?? null,
          model_window_seconds: modelInput.model_window_seconds ?? null,
          model_assumed_fps_after_resampling:
            modelInput.model_assumed_fps_after_resampling ?? null,
          window_metadata: modelInput.window_metadata ?? null,
          source_sample_count: modelInput.source_sample_count ?? null,
          source_duration_s: modelInput.source_duration_s ?? null,
          source_estimated_fps: modelInput.source_estimated_fps ?? null,
          preprocessing: modelInput.preprocessing ?? null,
        },
        notes: data.notes ?? [],
        omitted_from_compact_view: [
          "model_input.tensor",
          "model_input.values",
          "raw ROI sample buffer",
        ],
      };
    }

    async function renderModelPredictionSummaryFromServer({
      status = null,
      modelHr = null,
      spectralConsensus = null,
      modelDifference = null,
      greenSummary = null,
      posSummary = null,
      chromSummary = null,
      originalDurationS = null,
      usedDurationS = null,
      usedSamples = null,
      sourceFps = null,
      rawResponse = null,
    } = {}) {
      /*
      Fetch the server-rendered experimental model prediction panel.
      */

      await renderServerPartial({
        containerId: "live-model-prediction-output-container",
        routePath: "/ui/model-prediction-summary-json",
        method: "POST",
        params: {
          status: status,
          model_hr: modelHr,
          spectral_consensus: spectralConsensus,
          model_difference: modelDifference,
          green_summary: greenSummary,
          pos_summary: posSummary,
          chrom_summary: chromSummary,
          original_duration_s: originalDurationS,
          used_duration_s: usedDurationS,
          used_samples: usedSamples,
          source_fps: sourceFps,
          raw_response: rawResponse,
        },
        missingMessage: "Model prediction summary container not found.",
        errorMessage: "Could not render model prediction summary:",
      });
    }

    function getValidNumber(value) {
      if (value === null || value === undefined) {
        return null;
      }

      const numberValue = Number(value);

      if (Number.isNaN(numberValue)) {
        return null;
      }

      return numberValue;
    }

    function setElementText(id, text) {
      const element = document.getElementById(id);

      if (!element) {
        return;
      }

      element.textContent = text;
      element.innerText = text;
    }

    function setFinalInterpretationPanel({
      state = "idle",
      title = "No final estimate yet",
      detail = "Start a measurement. The app will use spectral HR as the primary estimate and treat model HR as experimental.",
      footnote = "Spectral HR remains primary.",
    } = {}) {
      const panel = document.getElementById("final-interpretation-panel");
      const stateEl = document.getElementById("final-interpretation-state");
      const detailEl = document.getElementById("final-interpretation-detail");
      const footnoteEl = document.getElementById("final-interpretation-footnote");

      if (!panel || !stateEl || !detailEl || !footnoteEl) {
        return;
      }

      const palette = {
        ready: {
          border: "#bbf7d0",
          background: "#f0fdf4",
          title: "#14532d",
          detail: "#166534",
          footnote: "#15803d",
        },
        warning: {
          border: "#fde68a",
          background: "#fffbeb",
          title: "#78350f",
          detail: "#92400e",
          footnote: "#b45309",
        },
        blocked: {
          border: "#fecaca",
          background: "#fef2f2",
          title: "#7f1d1d",
          detail: "#991b1b",
          footnote: "#b91c1c",
        },
        idle: {
          border: "#e2e8f0",
          background: "#ffffff",
          title: "#0f172a",
          detail: "#334155",
          footnote: "#64748b",
        },
      };

      const selected = palette[state] ?? palette.idle;

      panel.style.borderColor = selected.border;
      panel.style.background = selected.background;
      stateEl.style.color = selected.title;
      detailEl.style.color = selected.detail;
      footnoteEl.style.color = selected.footnote;

      stateEl.innerText = title;
      detailEl.innerText = detail;
      footnoteEl.innerText = footnote;
    }

    function updateFinalInterpretationFromCurrentDom() {
      const spectralHr = getElementTextTrimmed("spectral-consensus-summary", "Not analyzed yet");
      const modelHr = getElementTextTrimmed("live-model-hr-summary", "Not predicted yet");
      const modelDifference = getElementTextTrimmed(
        "model-spectral-difference-summary",
        "Not available"
      );
      const quality = getElementTextTrimmed("measurement-quality-summary", "Not analyzed yet");

      const spectralLower = spectralHr.toLowerCase();
      const modelLower = modelHr.toLowerCase();
      const qualityLower = quality.toLowerCase();

      const spectralHasBpm = spectralLower.includes("bpm");
      const signalAccepted = qualityLower.includes("accepted");
      const signalRejected =
        qualityLower.includes("rejected") ||
        spectralLower.includes("rejected") ||
        qualityLower.includes("failed") ||
        spectralLower.includes("failed");

      if (signalRejected) {
        setFinalInterpretationPanel({
          state: "blocked",
          title: "Do not use this measurement",
          detail:
            "The signal-quality gate rejected this measurement. Repeat the measurement with steadier face position and better lighting.",
          footnote: "Rejected measurements should not be interpreted as HR estimates.",
        });
        return;
      }

      if (!spectralHasBpm) {
        setFinalInterpretationPanel({
          state: "idle",
          title: "No final estimate yet",
          detail:
            "Start a measurement. The app will use spectral HR as the primary estimate and treat model HR as experimental.",
          footnote: "Spectral HR remains primary.",
        });
        return;
      }

      if (!signalAccepted) {
        setFinalInterpretationPanel({
          state: "warning",
          title: `Spectral estimate pending: ${spectralHr}`,
          detail:
            "A spectral value is visible, but the signal-quality state is not accepted yet. Wait for the full measurement result.",
          footnote: "Do not over-interpret partial results.",
        });
        return;
      }

      if (modelLower.includes("disagrees")) {
        setFinalInterpretationPanel({
          state: "warning",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            `Signal accepted. Experimental model disagreed by ${modelDifference}, ` +
            "so model HR is not used as the primary estimate.",
          footnote: "Spectral HR remains primary because model agreement failed.",
        });
        return;
      }

      if (modelLower.includes("skipped")) {
        setFinalInterpretationPanel({
          state: "ready",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            "Signal accepted. Experimental model was skipped by a guardrail, so spectral HR remains the app estimate.",
          footnote: "Model skip is a safety state, not a failed spectral measurement.",
        });
        return;
      }

      if (modelLower.includes("not run")) {
        setFinalInterpretationPanel({
          state: "ready",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            "Signal accepted. Experimental model was not run for this measurement, so spectral HR remains the app estimate.",
          footnote: "Spectral HR remains primary.",
        });
        return;
      }

      if (modelLower.includes("rejected")) {
        setFinalInterpretationPanel({
          state: "ready",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            "Signal accepted. Experimental model-window quality rejected the model input, so spectral HR remains primary.",
          footnote: "Model rejection does not invalidate an accepted spectral estimate.",
        });
        return;
      }

      if (modelLower.includes("unavailable") || modelLower.includes("failed")) {
        setFinalInterpretationPanel({
          state: "ready",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            "Signal accepted. Experimental model output is unavailable for this run, so spectral HR remains primary.",
          footnote: "Model availability is separate from spectral signal quality.",
        });
        return;
      }

      if (modelLower.includes("not predicted")) {
        setFinalInterpretationPanel({
          state: "ready",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            "Signal accepted. Experimental model has not been run yet; spectral HR is the current app estimate.",
          footnote: "Spectral HR remains primary.",
        });
        return;
      }

      if (modelLower.includes("bpm")) {
        setFinalInterpretationPanel({
          state: "ready",
          title: `Use spectral estimate: ${spectralHr}`,
          detail:
            `Signal accepted. Experimental model returned ${modelHr}; use it only as a secondary comparison.`,
          footnote: "Spectral HR remains primary even when the model returns a value.",
        });
        return;
      }

      setFinalInterpretationPanel({
        state: "ready",
        title: `Use spectral estimate: ${spectralHr}`,
        detail:
          "Signal accepted. Spectral HR is the app estimate; model output remains experimental.",
        footnote: "Spectral HR remains primary.",
      });
    }

    function resetFinalInterpretationPanel() {
      setFinalInterpretationPanel();
    }

    function clearCanvasById(id) {
      const canvasEl = document.getElementById(id);

      if (!canvasEl) {
        return;
      }

      const context = canvasEl.getContext("2d");

      if (!context) {
        return;
      }

      context.clearRect(0, 0, canvasEl.width, canvasEl.height);
    }

    function resetCapturedFrameDiagnostics() {
      hasCapturedFrame = false;
      clearCanvasById("snapshot-canvas");
      setElementText("backend-frame-debug", "No frame sent to backend yet.");
      setElementText("backend-face-debug", "No face detection request sent yet.");
    }


    function resetMeasurementOutputs(expectedRevision = null) {
      latestRoiAnalysisDisplayState = null;

      setElementText("spectral-consensus-summary", "Not analyzed yet");
      setElementText("live-model-hr-summary", "Not predicted yet");
      setElementText("model-spectral-difference-summary", "Not predicted yet");
      setElementText("measurement-quality-summary", "Not analyzed yet");
      setElementText("measurement-quality-detail", "Spectral signal-quality gate");
      resetFinalInterpretationPanel();

      renderMeasurementResultCardsPlaceholderFromServer({
        expectedRevision: expectedRevision,
      });
      renderSignalSummaryCardsPlaceholderFromServer({
        expectedRevision: expectedRevision,
      });

      setElementText("roi-sampling-summary", "No ROI samples collected yet.");
      renderRoiAnalysisSummaryPlaceholderFromServer({
        expectedRevision: expectedRevision,
      });
      renderModelPredictionSummaryPlaceholderFromServer({
        expectedRevision: expectedRevision,
      });
      setElementText("backend-frame-debug", "No frame sent to backend yet.");
      setElementText("backend-face-debug", "No face detection request sent yet.");

      clearCanvasById("main-pulse-wave-canvas");
      clearCanvasById("roi-green-trace-canvas");
      clearCanvasById("roi-green-normalized-trace-canvas");
      resetMeasurementProgress();
      updateDemoReadinessPanel();
    }

    function setMeasurementProgress(progressFraction, progressText) {
      const progressBar = document.getElementById("measurement-progress-bar");
      const progressTextEl = document.getElementById("measurement-progress-text");

      const safeProgress = Math.max(0, Math.min(1, Number(progressFraction) || 0));
      const percentText = `${Math.round(safeProgress * 100)}%`;

      if (progressBar) {
        progressBar.style.width = percentText;
      }

      if (progressTextEl) {
        progressTextEl.innerText = progressText ?? percentText;
      }
    }

    function resetMeasurementRunState(activeRevision, { resetCapturedFrame = false } = {}) {
      const startButton = document.getElementById("start-roi-sampling-button");

      if (mainMeasurementTimer !== null) {
        clearTimeout(mainMeasurementTimer);
        mainMeasurementTimer = null;
      }

      roiSamplingRunId += 1;
      stopMeasurementProgressTimer();

      clearRoiSamplingSchedule();

      roiSamples = [];
      roiSamplingStartMs = null;
      roiSamplingInFlight = false;

      /*
      Do not clear window.livePredictionRuns here.
      The repeatability table compares predictions across repeated measurements
      during the same browser page session.
      */

      if (startButton) {
        startButton.disabled = false;
        startButton.innerText = "Start ROI sampling";
      }

      resetMeasurementOutputs(activeRevision);

      if (resetCapturedFrame) {
        resetCapturedFrameDiagnostics();
      }

      drawMainPulseWaveformPlaceholder();
    }

    function setMeasurementStatus(summary, detail = "") {
      setElementText("measurement-status-summary", summary);
      setElementText("measurement-status-detail", detail);
    }

    function resetMeasurementProgress() {
      setMeasurementStatus(
        "Ready.",
        "Start the camera, then start a measurement while holding still."
      );
      setMeasurementProgress(0.0, "0%");
    }

    function stopMeasurementProgressTimer() {
      if (mainMeasurementProgressTimer !== null) {
        clearInterval(mainMeasurementProgressTimer);
        mainMeasurementProgressTimer = null;
      }
    }

    function startMeasurementProgressTimer(activeRevision, startedAtMs) {
      stopMeasurementProgressTimer();

      mainMeasurementProgressTimer = setInterval(() => {
        if (activeRevision !== measurementRevision || !mainMeasurementInProgress) {
          stopMeasurementProgressTimer();
          return;
        }

        const elapsedMs = performance.now() - startedAtMs;
        const progress = elapsedMs / MAIN_MEASUREMENT_DURATION_MS;
        const totalSeconds = MAIN_MEASUREMENT_DURATION_MS / 1000.0;
        const elapsedSeconds = Math.min(
          totalSeconds,
          Math.max(0.0, elapsedMs / 1000.0)
        );
        const remainingSeconds = Math.max(0.0, totalSeconds - elapsedSeconds);

        const phaseText = elapsedSeconds < 3.0
          ? "Stabilizing signal. Hold still."
          : "Collecting rPPG signal. Keep your face steady.";

        if (isMobileDemoViewport()) {
          setMeasurementStatus(
            "Measuring...",
            `${phaseText} ${remainingSeconds.toFixed(1)} s remaining.`
          );
          setMeasurementProgress(
            progress,
            `${remainingSeconds.toFixed(1)} s remaining`
          );
          return;
        }

        setMeasurementStatus(
          "Measuring...",
          `${phaseText} ${elapsedSeconds.toFixed(1)} / ${totalSeconds.toFixed(1)} s`
        );
        setMeasurementProgress(
          progress,
          `${elapsedSeconds.toFixed(1)} / ${totalSeconds.toFixed(1)} s`
        );
      }, 250);
    }


    function hasPrimaryStateToClear() {
      const predictionRunCount = Array.isArray(window.livePredictionRuns)
        ? window.livePredictionRuns.length
        : 0;

      return (
        hasCapturedFrame ||
        roiSamples.length > 0 ||
        predictionRunCount > 0 ||
        roiSamplingScheduleId !== null ||
        mainMeasurementInProgress
      );
    }

    function hasEnoughRoiSamplesForBackend() {
      return roiSamples.length >= 20;
    }

    function updateAdvancedDiagnosticStatus({
      cameraActive,
      measurementActive,
      samplingActive,
      enoughRoiSamples,
    }) {
      if (measurementActive) {
        setElementText(
          "advanced-manual-controls-status",
          "Main measurement is running. Manual diagnostic controls are paused."
        );
        setElementText(
          "advanced-frame-controls-status",
          "Frame diagnostics are paused during the main measurement."
        );
        return;
      }

      if (!cameraActive) {
        setElementText(
          "advanced-manual-controls-status",
          "Start the camera to enable manual ROI sampling."
        );
        setElementText(
          "advanced-frame-controls-status",
          "Start the camera to enable frame capture."
        );
        return;
      }

      if (samplingActive) {
        setElementText(
          "advanced-manual-controls-status",
          `ROI sampling active. Collected ${roiSamples.length} sample(s). Stop sampling before analysis or model prediction.`
        );
      } else if (enoughRoiSamples) {
        setElementText(
          "advanced-manual-controls-status",
          `Collected ${roiSamples.length} sample(s). Analysis and model prediction are available.`
        );
      } else if (roiSamples.length > 0) {
        setElementText(
          "advanced-manual-controls-status",
          `Collected ${roiSamples.length} / 20 sample(s). Continue sampling before analysis or model prediction.`
        );
      } else {
        setElementText(
          "advanced-manual-controls-status",
          "Camera is ready. Start ROI sampling to collect diagnostic samples."
        );
      }

      if (hasCapturedFrame) {
        setElementText(
          "advanced-frame-controls-status",
          "Captured frame is ready. You can send it to the backend or run face / ROI overlay diagnostics."
        );
      } else {
        setElementText(
          "advanced-frame-controls-status",
          "Camera is ready. Capture one frame to enable frame diagnostics."
        );
      }
    }
    
    function setDemoReadinessItem(valueId, detailId, state, value, detail) {
      const valueEl = document.getElementById(valueId);
      const detailEl = document.getElementById(detailId);

      if (!valueEl || !detailEl) {
        return;
      }

      const palette = {
        ready: { background: "#dcfce7", color: "#166534" },
        busy: { background: "#dbeafe", color: "#1d4ed8" },
        warning: { background: "#fef3c7", color: "#92400e" },
        blocked: { background: "#fee2e2", color: "#991b1b" },
        idle: { background: "#e2e8f0", color: "#334155" },
      };

      const selected = palette[state] ?? palette.idle;

      valueEl.innerText = value;
      valueEl.style.background = selected.background;
      valueEl.style.color = selected.color;
      detailEl.innerText = detail;
    }

    function getElementTextTrimmed(id, fallback = "") {
      const element = document.getElementById(id);

      if (!element) {
        return fallback;
      }

      const text = (element.textContent ?? element.innerText ?? "").trim();

      return text.length > 0 ? text : fallback;
    }

    function computeRoiSampleDurationS(samples) {
      if (!Array.isArray(samples) || samples.length < 2) {
        return null;
      }

      const firstTime = getValidNumber(samples[0]?.t_s);
      const lastTime = getValidNumber(samples[samples.length - 1]?.t_s);

      if (firstTime === null || lastTime === null || lastTime <= firstTime) {
        return null;
      }

      return lastTime - firstTime;
    }

    function updateDemoReadinessPanel() {
      const panel = document.getElementById("demo-readiness-panel");

      if (!panel) {
        return;
      }

      const cameraActive = Boolean(cameraStream);
      const measurementActive = Boolean(mainMeasurementInProgress);
      const samplingActive = roiSamplingScheduleId !== null;
      const enoughSamples = roiSamples.length >= 20;
      const secureContext = Boolean(window.isSecureContext);

      setDemoReadinessItem(
        "readiness-secure-context-value",
        "readiness-secure-context-detail",
        secureContext ? "ready" : "blocked",
        secureContext ? "Secure" : "Blocked",
        secureContext
          ? "Camera APIs should be available in this browser context."
          : "Browser security may block the camera. Use HTTPS or localhost."
      );

      setDemoReadinessItem(
        "readiness-camera-value",
        "readiness-camera-detail",
        cameraActive ? (measurementActive ? "busy" : "ready") : "idle",
        cameraActive ? (measurementActive ? "Measuring" : "Camera ready") : "Not started",
        cameraActive
          ? "Camera preview is open. Frames are processed only during requested actions."
          : "Camera is closed."
      );

      const durationS = computeRoiSampleDurationS(roiSamples);
      const sampleDetail =
        durationS === null
          ? "No measurement window has been collected yet."
          : `${roiSamples.length} sample(s) over ${durationS.toFixed(2)} s.`;

      setDemoReadinessItem(
        "readiness-samples-value",
        "readiness-samples-detail",
        enoughSamples ? "ready" : samplingActive ? "busy" : roiSamples.length > 0 ? "warning" : "idle",
        `${roiSamples.length} sample(s)`,
        enoughSamples
          ? `${sampleDetail} Backend analysis can run.`
          : `${sampleDetail} At least 20 samples are needed before analysis.`
      );

      const effectiveModelFps = estimateRoiSamplesEffectiveFpsForModel(roiSamples, 12.0);

      let fpsState = "idle";
      let fpsValue = "Waiting";
      let fpsDetail = "Experimental model preprocessing needs at least 8.0 Hz source sampling.";

      if (effectiveModelFps !== null) {
        fpsValue = `${effectiveModelFps.toFixed(2)} Hz`;

        if (effectiveModelFps >= 12.0) {
          fpsState = "ready";
          fpsDetail = "Good live sampling margin for the experimental model path.";
        } else if (effectiveModelFps >= 8.0) {
          fpsState = "warning";
          fpsDetail = "Usable for the model path, but with limited sampling margin.";
        } else {
          fpsState = "blocked";
          fpsDetail = "Too low for training-style model preprocessing; model should be skipped.";
        }
      }

      setDemoReadinessItem(
        "readiness-model-fps-value",
        "readiness-model-fps-detail",
        fpsState,
        fpsValue,
        fpsDetail
      );

      const signalText = getElementTextTrimmed("measurement-quality-summary", "Not analyzed yet");
      const signalDetail = getElementTextTrimmed(
        "measurement-quality-detail",
        "Full-buffer spectral quality has not been computed yet."
      );

      const signalState = signalText.toLowerCase().includes("accepted")
        ? "ready"
        : signalText.toLowerCase().includes("rejected")
          ? "blocked"
          : signalText.toLowerCase().includes("failed")
            ? "blocked"
            : "idle";

      setDemoReadinessItem(
        "readiness-signal-value",
        "readiness-signal-detail",
        signalState,
        signalText,
        signalDetail
      );

      const modelText = getElementTextTrimmed("live-model-hr-summary", "Not predicted yet");
      const modelDifferenceText = getElementTextTrimmed(
        "model-spectral-difference-summary",
        "Not available"
      );
      const modelTextLower = modelText.toLowerCase();

      let modelState = "idle";
      let modelDetail = "Experimental only. Spectral HR remains the primary app estimate.";

      if (modelTextLower.includes("bpm")) {
        modelState = "ready";
        modelDetail = `Model returned ${modelText}. Spectral HR remains the primary app estimate.`;
      } else if (modelTextLower.includes("disagrees")) {
        modelState = "warning";
        modelDetail =
          `Model differs from model-window spectral by ${modelDifferenceText}. ` +
          "Keep spectral HR primary.";
      } else if (modelTextLower.includes("rejected")) {
        modelState = "warning";
        modelDetail =
          "Model-window quality gate rejected this input. Keep spectral HR primary.";
      } else if (modelTextLower.includes("skipped")) {
        modelState = "warning";
        modelDetail =
          "Model was skipped by a guardrail. Keep spectral HR primary.";
      } else if (modelTextLower.includes("unavailable") || modelTextLower.includes("not run")) {
        modelState = "warning";
        modelDetail =
          "Model output is unavailable for this run. Keep spectral HR primary.";
      } else if (modelTextLower.includes("failed")) {
        modelState = "blocked";
        modelDetail =
          "Model prediction failed. Keep spectral HR primary and inspect diagnostics if needed.";
      }

      setDemoReadinessItem(
        "readiness-model-value",
        "readiness-model-detail",
        modelState,
        modelText,
        modelDetail
      );
    }


    function isMobileDemoViewport() {
      return window.matchMedia("(max-width: 767px)").matches;
    }

    function waitForVideoPreviewReady(timeoutMs = 4000) {
      const videoEl = document.getElementById("camera-video");

      if (!videoEl) {
        return Promise.resolve(false);
      }

      if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
        return Promise.resolve(true);
      }

      return new Promise(resolve => {
        let settled = false;

        const cleanup = () => {
          videoEl.removeEventListener("loadedmetadata", checkReady);
          videoEl.removeEventListener("canplay", checkReady);
          clearInterval(intervalId);
          clearTimeout(timeoutId);
        };

        const finish = value => {
          if (settled) {
            return;
          }

          settled = true;
          cleanup();
          resolve(value);
        };

        const checkReady = () => {
          if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
            finish(true);
          }
        };

        const intervalId = setInterval(checkReady, 100);
        const timeoutId = setTimeout(() => finish(false), timeoutMs);

        videoEl.addEventListener("loadedmetadata", checkReady);
        videoEl.addEventListener("canplay", checkReady);

        checkReady();
      });
    }

    function ensureMobileCameraGuidanceOverlay() {
      const videoEl = document.getElementById("camera-video");

      if (!videoEl) {
        return null;
      }

      let overlayEl = document.getElementById("mobile-camera-guidance-overlay");

      if (overlayEl) {
        return overlayEl;
      }

      const parentEl = videoEl.parentElement;

      if (!parentEl) {
        return null;
      }

      let frameEl = document.getElementById("mobile-camera-preview-frame");

      if (!frameEl) {
        frameEl = document.createElement("div");
        frameEl.id = "mobile-camera-preview-frame";

        parentEl.insertBefore(frameEl, videoEl);
        frameEl.appendChild(videoEl);
      }

      overlayEl = document.createElement("div");
      overlayEl.id = "mobile-camera-guidance-overlay";
      overlayEl.setAttribute("data-visible", "false");

      overlayEl.innerHTML = `
        <div>
          <div id="mobile-camera-guidance-primary">Position your face</div>
          <div id="mobile-camera-guidance-secondary">Place your head in the middle of the camera view.</div>
        </div>
      `;

      frameEl.appendChild(overlayEl);

      return overlayEl;
    }

    function setMobileCameraGuidanceOverlay(primaryText, secondaryText = "", visible = true) {
      const overlayEl = ensureMobileCameraGuidanceOverlay();

      if (!overlayEl) {
        return;
      }

      const primaryEl = document.getElementById("mobile-camera-guidance-primary");
      const secondaryEl = document.getElementById("mobile-camera-guidance-secondary");

      if (primaryEl) {
        primaryEl.innerText = primaryText;
      }

      if (secondaryEl) {
        secondaryEl.innerText = secondaryText;
      }

      overlayEl.setAttribute("data-visible", visible ? "true" : "false");
    }

    function hideMobileCameraGuidanceOverlay() {
      const overlayEl = document.getElementById("mobile-camera-guidance-overlay");

      if (overlayEl) {
        overlayEl.setAttribute("data-visible", "false");
      }
    }

    function runMobileMeasurementPrepareCountdown(activeRevision, prepareSeconds = 3) {
      if (!isMobileDemoViewport()) {
        return Promise.resolve(true);
      }

      return new Promise(resolve => {
        let settled = false;
        let remainingSeconds = prepareSeconds;
        let intervalId = null;
        let startTimeoutId = null;

        const finish = value => {
          if (settled) {
            return;
          }

          settled = true;

          if (intervalId !== null) {
            clearInterval(intervalId);
          }

          if (startTimeoutId !== null) {
            clearTimeout(startTimeoutId);
          }

          if (!value) {
            hideMobileCameraGuidanceOverlay();
          }

          resolve(value);
        };

        const renderStep = () => {
          if (activeRevision !== measurementRevision || !mainMeasurementInProgress) {
            finish(false);
            return;
          }

          if (remainingSeconds > 0) {
            setMobileCameraGuidanceOverlay(
              "Position your face",
              `Place your head in the middle. Measurement starts in ${remainingSeconds}.`,
              true
            );
            setMeasurementStatus(
              "Get ready.",
              `Place your head in the middle of the camera view. Measurement starts in ${remainingSeconds}.`
            );
            setMeasurementProgress(0.0, `Starts in ${remainingSeconds}`);

            remainingSeconds -= 1;
            return;
          }

          setMobileCameraGuidanceOverlay(
            "Start",
            "Hold still and keep your face centered.",
            true
          );
          setMeasurementStatus(
            "Measuring...",
            "Start. Hold still and keep your face centered."
          );
          setMeasurementProgress(0.0, "15.0 s remaining");

          if (intervalId !== null) {
            clearInterval(intervalId);
            intervalId = null;
          }

          startTimeoutId = setTimeout(() => {
            hideMobileCameraGuidanceOverlay();
            finish(true);
          }, 450);
        };

        renderStep();
        intervalId = setInterval(renderStep, 1000);
      });
    }

    function scrollToMeasurementResultsOnMobile() {
      if (!isMobileDemoViewport()) {
        return;
      }

      const finalInterpretationEl = document.getElementById("final-interpretation-panel");
      const fallbackResultsEl = document.getElementById("measurement-result-cards-container");
      const scrollTarget = finalInterpretationEl ?? fallbackResultsEl;

      if (scrollTarget) {
        scrollTarget.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    }

    function updatePrimaryControlButtons() {
      const cameraActive = Boolean(cameraStream);
      const measurementActive = Boolean(mainMeasurementInProgress);
      const samplingActive = roiSamplingScheduleId !== null;
      const clearable = hasPrimaryStateToClear();
      const enoughRoiSamples = hasEnoughRoiSamplesForBackend();

      const startCameraButton = document.getElementById("start-camera-button");
      const startMeasurementButton = document.getElementById("start-measurement-button");
      const stopMeasurementButton = document.getElementById("stop-measurement-button");
      const stopCameraButton = document.getElementById("stop-camera-button");
      const clearButton = document.getElementById("clear-roi-samples-button");

      const captureFrameButton = document.getElementById("capture-frame-button");
      const sendFrameButton = document.getElementById("send-frame-button");
      const detectFaceButton = document.getElementById("detect-face-button");
      const startRoiSamplingButton = document.getElementById("start-roi-sampling-button");
      const stopRoiSamplingButton = document.getElementById("stop-roi-sampling-button");
      const analyzeRoiSeriesButton = document.getElementById("analyze-roi-series-button");
      const runLiveModelButton = document.getElementById("run-live-model-button");

      if (startCameraButton) {
        startCameraButton.disabled = cameraActive;
        startCameraButton.innerText = cameraActive ? "Camera started" : "Start camera";
      }

      if (startMeasurementButton) {
        startMeasurementButton.disabled = measurementActive || (!cameraActive && !isMobileDemoViewport());
        startMeasurementButton.innerText = measurementActive ? "Measuring..." : "Start measurement";
      }

      if (stopMeasurementButton) {
        stopMeasurementButton.disabled = !measurementActive;
      }

      if (stopCameraButton) {
        stopCameraButton.disabled = !cameraActive;
      }

      if (clearButton) {
        clearButton.disabled = measurementActive || !clearable;
      }

      if (captureFrameButton) {
        captureFrameButton.disabled = !cameraActive || measurementActive;
      }

      if (sendFrameButton) {
        sendFrameButton.disabled = !hasCapturedFrame || measurementActive;
      }

      if (detectFaceButton) {
        detectFaceButton.disabled = !hasCapturedFrame || measurementActive;
      }

      if (startRoiSamplingButton) {
        startRoiSamplingButton.disabled = !cameraActive || samplingActive || measurementActive;
        startRoiSamplingButton.innerText = samplingActive ? "Sampling..." : "Start ROI sampling";
      }

      if (stopRoiSamplingButton) {
        stopRoiSamplingButton.disabled = !samplingActive || measurementActive;
      }

      if (analyzeRoiSeriesButton) {
        analyzeRoiSeriesButton.disabled = !enoughRoiSamples || samplingActive || measurementActive;
      }

      if (runLiveModelButton) {
        runLiveModelButton.disabled = !enoughRoiSamples || samplingActive || measurementActive;
      }

      updateAdvancedDiagnosticStatus({
        cameraActive,
        measurementActive,
        samplingActive,
        enoughRoiSamples,
      });

      updateDemoReadinessPanel();
    }


    function setMainMeasurementButtonsState(isRunning) {
      mainMeasurementInProgress = Boolean(isRunning);
      updatePrimaryControlButtons();
    }

    function mean(values) {
      if (values.length === 0) {
        return null;
      }

      return values.reduce((sum, value) => sum + value, 0) / values.length;
    }


    function finiteValues(values) {
      return values
        .filter(value => value !== null && value !== undefined)
        .map(value => Number(value))
        .filter(value => Number.isFinite(value));
    }

    function meanFinite(values) {
      const numericValues = finiteValues(values);

      if (numericValues.length === 0) {
        return null;
      }

      return mean(numericValues);
    }

    function formatBpm(value) {
      const numberValue = getValidNumber(value);

      if (numberValue === null) {
        return "none";
      }

      return `${numberValue.toFixed(1)} bpm`;
    }

    function formatSignedBpm(value) {
      const numberValue = getValidNumber(value);

      if (numberValue === null) {
        return "none";
      }

      const sign = numberValue >= 0 ? "+" : "";

      return `${sign}${numberValue.toFixed(1)} bpm`;
    }

    function formatNumber(value, digits = 2) {
      const numberValue = getValidNumber(value);

      if (numberValue === null) {
        return "none";
      }

      return numberValue.toFixed(digits);
    }

    function getRoiColor(name) {
      return ROI_COLORS[name] ?? "#2563eb";
    }

    async function parseJsonResponseEvenOnError(response) {
      let data = null;

      try {
        data = await response.json();
      } catch (error) {
        data = {
          status: "error",
          message: `Could not parse JSON response: ${error}`
        };
      }

      if (!response.ok) {
        const message = data.message ?? `HTTP ${response.status}`;
        const exceptionType = data.exception_type ?? "BackendError";
        throw new Error(`${response.status} ${exceptionType}: ${message}`);
      }

      return data;
    }

    async function startCameraPreview() {
      const videoEl = document.getElementById("camera-video");
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-camera-button");

      if (cameraStream) {
        updatePrimaryControlButtons();
        return true;
      }

      if (startButton) {
        startButton.disabled = true;
        startButton.innerText = "Starting...";
      }

      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          },
          audio: false
        });

        videoEl.srcObject = cameraStream;

        if (statusEl) {
          statusEl.innerText =
            "Camera started. Preview is local in the browser. Frames are not sent to the backend automatically.";
        }

        updatePrimaryControlButtons();
        return true;
      } catch (error) {
        if (statusEl) {
          statusEl.innerText = `Camera start failed: ${error}`;
        }

        updatePrimaryControlButtons();
        return false;
      } finally {
        if (startButton) {
          startButton.disabled = false;
          startButton.innerText = "Start camera";
        }

        updatePrimaryControlButtons();
      }
    }

    function stopCameraPreview() {
      const videoEl = document.getElementById("camera-video");
      const statusEl = document.getElementById("camera-status");
      const startSamplingButton = document.getElementById("start-roi-sampling-button");

      if (mainMeasurementTimer !== null) {
        clearTimeout(mainMeasurementTimer);
        mainMeasurementTimer = null;
      }

      setMainMeasurementButtonsState(false);
      measurementRevision += 1;
      stopMeasurementProgressTimer();
      setMeasurementStatus("Camera stopped.", "Start the camera again before running a new measurement.");

      clearRoiSamplingSchedule();

      if (startSamplingButton) {
        startSamplingButton.disabled = false;
        startSamplingButton.innerText = "Start ROI sampling";
      }

      if (cameraStream) {
        const tracks = cameraStream.getTracks();

        for (const track of tracks) {
          track.stop();
        }

        cameraStream = null;
      }

      if (videoEl) {
        videoEl.srcObject = null;
      }

      if (statusEl) {
        statusEl.innerText = "Camera stopped. Captured frame remains local in the canvas.";
      }

      updatePrimaryControlButtons();
    }

    function captureOneFrame() {
      const videoEl = document.getElementById("camera-video");
      const canvasEl = document.getElementById("snapshot-canvas");
      const statusEl = document.getElementById("camera-status");

      if (!cameraStream) {
        statusEl.innerText = "Cannot capture frame: camera is not started.";
        updatePrimaryControlButtons();
        return false;
      }

      const width = videoEl.videoWidth;
      const height = videoEl.videoHeight;

      if (width === 0 || height === 0) {
        statusEl.innerText = "Cannot capture frame yet: video dimensions are not ready.";
        updatePrimaryControlButtons();
        return false;
      }

      canvasEl.width = width;
      canvasEl.height = height;

      const context = canvasEl.getContext("2d");
      context.drawImage(videoEl, 0, 0, width, height);

      hasCapturedFrame = true;

      statusEl.innerText =
        `Captured one local frame: ${width} x ${height}. Frame was not sent to backend.`;

      updatePrimaryControlButtons();

      return true;
    }

    function getCapturedFrameDataUrl() {
      const canvasEl = document.getElementById("snapshot-canvas");

      if (!hasCapturedFrame) {
        throw new Error("Capture one frame first.");
      }

      return canvasEl.toDataURL("image/jpeg", 0.85);
    }

    async function sendCapturedFrameToBackend() {
      const statusEl = document.getElementById("camera-status");
      const debugEl = document.getElementById("backend-frame-debug");
      const sendButton = document.getElementById("send-frame-button");

      sendButton.disabled = true;
      sendButton.innerText = "Sending...";

      try {
        const imageDataUrl = getCapturedFrameDataUrl();

        const response = await fetch("/api/debug-frame", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            image_data_url: imageDataUrl
          })
        });

        const data = await parseJsonResponseEvenOnError(response);

        debugEl.innerText = JSON.stringify(data, null, 2);

        statusEl.innerText =
          `Backend decoded frame: ${data.frame.width} x ${data.frame.height}, ` +
          `${data.frame.channels} channels. Frame was not stored.`;
      } catch (error) {
        debugEl.innerText = `Frame send failed: ${error}`;
        statusEl.innerText = `Frame send failed: ${error}`;
      } finally {
        sendButton.disabled = false;
        sendButton.innerText = "Send frame to backend";
      }
    }

    function getLabelPosition(canvasEl, box, label, fontSize) {
      const padding = 4;
      const labelHeight = fontSize + 8;

      const context = canvasEl.getContext("2d");
      const labelWidth = context.measureText(label).width + 12;

      let labelX = box.x_min;
      let labelY = box.y_min - labelHeight - padding;

      if (labelY < 0) {
        labelY = box.y_max + padding;
      }

      if (labelY + labelHeight > canvasEl.height) {
        labelY = Math.max(0, box.y_min + padding);
      }

      if (labelX + labelWidth > canvasEl.width) {
        labelX = canvasEl.width - labelWidth - padding;
      }

      if (labelX < 0) {
        labelX = padding;
      }

      return {
        x: labelX,
        y: labelY,
        width: labelWidth,
        height: labelHeight
      };
    }

    function drawRectangleWithLabel(context, canvasEl, box, label, strokeStyle) {
      context.save();

      context.strokeStyle = strokeStyle;
      context.lineWidth = 3;
      context.strokeRect(box.x_min, box.y_min, box.width, box.height);

      const fontSize = 13;
      context.font = `${fontSize}px sans-serif`;

      const labelPosition = getLabelPosition(canvasEl, box, label, fontSize);

      context.fillStyle = "rgba(15, 23, 42, 0.88)";
      context.fillRect(
        labelPosition.x,
        labelPosition.y,
        labelPosition.width,
        labelPosition.height
      );

      context.fillStyle = "white";
      context.fillText(
        label,
        labelPosition.x + 6,
        labelPosition.y + fontSize + 2
      );

      context.restore();
    }

    function drawFaceAndRoiOverlay(data) {
      const canvasEl = document.getElementById("snapshot-canvas");
      const context = canvasEl.getContext("2d");

      const faceDebug = data.face_debug;

      if (!faceDebug || !faceDebug.face_detected) {
        return;
      }

      const faceBox = faceDebug.face?.bbox;
      const roiDebug = faceDebug.roi_debug;
      const rois = roiDebug?.rois ?? [];

      if (faceBox) {
        drawRectangleWithLabel(context, canvasEl, faceBox, "face", "#0f172a");
      }

      for (const roi of rois) {
        if (!roi.usable) {
          continue;
        }

        const label = roi.name;
        const color = getRoiColor(roi.name);

        drawRectangleWithLabel(context, canvasEl, roi.box, label, color);
      }
    }

    function summarizeRoiMeans(data) {
      const rois = data.face_debug?.roi_debug?.rois ?? [];

      if (rois.length === 0) {
        return "No ROI summaries returned.";
      }

      const parts = [];

      for (const roi of rois) {
        const meanRgb = roi.rgb_summary?.mean_rgb;
        const qualityStatus = roi.quality?.status ?? "unknown";

        if (!meanRgb) {
          parts.push(`${roi.name}: no RGB summary`);
          continue;
        }

        parts.push(
          `${roi.name}: ${qualityStatus}, ` +
          `R=${meanRgb.r.toFixed(1)}, G=${meanRgb.g.toFixed(1)}, B=${meanRgb.b.toFixed(1)}`
        );
      }

      return parts.join(" | ");
    }

    async function detectFaceInBackend() {
      const statusEl = document.getElementById("camera-status");
      const debugEl = document.getElementById("backend-face-debug");
      const detectButton = document.getElementById("detect-face-button");

      detectButton.disabled = true;
      detectButton.innerText = "Detecting...";

      try {
        const imageDataUrl = getCapturedFrameDataUrl();

        const response = await fetch("/api/debug-face", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            image_data_url: imageDataUrl
          })
        });

        const data = await parseJsonResponseEvenOnError(response);

        debugEl.innerText = JSON.stringify(data, null, 2);

        const faceDetected = data.face_debug?.face_detected;
        const landmarkCount = data.face_debug?.face?.landmark_count;
        const bbox = data.face_debug?.face?.bbox;

        if (faceDetected) {
          drawFaceAndRoiOverlay(data);

          const roiSummary = summarizeRoiMeans(data);

          statusEl.innerText =
            `Backend detected face: ${landmarkCount} landmarks, bbox ${bbox.width} x ${bbox.height}. ` +
            `ROIs drawn. ${roiSummary}`;
        } else {
          statusEl.innerText = "Backend did not detect a face in this frame.";
        }
      } catch (error) {
        debugEl.innerText = `Face detection failed: ${error}`;
        statusEl.innerText = `Face detection failed: ${error}`;
      } finally {
        detectButton.disabled = false;
        detectButton.innerText = "Detect face + draw ROIs";
      }
    }

    function getOrCreateHiddenSamplingCanvas() {
      let canvasEl = document.getElementById("sampling-hidden-canvas");

      if (canvasEl) {
        return canvasEl;
      }

      canvasEl = document.createElement("canvas");
      canvasEl.id = "sampling-hidden-canvas";
      canvasEl.width = 640;
      canvasEl.height = 480;
      canvasEl.style.display = "none";

      document.body.appendChild(canvasEl);

      return canvasEl;
    }

    function captureFrameForSamplingDataUrl() {
      const videoEl = document.getElementById("camera-video");
      const statusEl = document.getElementById("camera-status");
      const canvasEl = getOrCreateHiddenSamplingCanvas();

      if (!cameraStream) {
        statusEl.innerText = "Cannot capture sampling frame: camera is not started.";
        return null;
      }

      const sourceWidth = videoEl.videoWidth;
      const sourceHeight = videoEl.videoHeight;

      if (sourceWidth === 0 || sourceHeight === 0) {
        statusEl.innerText = "Cannot capture sampling frame yet: video dimensions are not ready.";
        return null;
      }

      const SAMPLING_TARGET_WIDTH = 640;

      const scale =
        SAMPLING_TARGET_WIDTH && sourceWidth > SAMPLING_TARGET_WIDTH
          ? SAMPLING_TARGET_WIDTH / sourceWidth
          : 1.0;

      const width = Math.round(sourceWidth * scale);
      const height = Math.round(sourceHeight * scale);

      canvasEl.width = width;
      canvasEl.height = height;

      const context = canvasEl.getContext("2d");
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(videoEl, 0, 0, width, height);

      return canvasEl.toDataURL("image/jpeg", 0.85);
    }

    function extractRoiSampleFromBackendResponse(data, acquisitionTiming = {}) {
      const nowMs = performance.now();
      const elapsedS =
        roiSamplingStartMs === null
          ? 0.0
          : (nowMs - roiSamplingStartMs) / 1000.0;

      const faceDebug = data.face_debug;
      const roiDebug = faceDebug?.roi_debug;
      const rois = roiDebug?.rois ?? [];

      const sample = {
        t_s: elapsedS,
        face_detected: Boolean(faceDebug?.face_detected),
        roi_quality_overall: roiDebug?.quality_summary?.overall_status ?? "unknown",
        acquisition_timing: {
          capture_to_data_url_ms: getValidNumber(acquisitionTiming.captureToDataUrlMs),
          request_round_trip_ms: getValidNumber(acquisitionTiming.requestRoundTripMs),
          response_parse_ms: getValidNumber(acquisitionTiming.responseParseMs),
          total_browser_sample_ms: getValidNumber(acquisitionTiming.totalBrowserSampleMs),
          backend_total_request_processing_ms:
            getValidNumber(data?.timing_ms?.total_request_processing_ms),
          backend_face_debug_total_ms:
            getValidNumber(data?.timing_ms?.face_debug_total_ms),
          backend_decode_ms: getValidNumber(data?.timing_ms?.decode_ms),
          http_status: Number.isInteger(acquisitionTiming.httpStatus)
            ? acquisitionTiming.httpStatus
            : null
        },
        rois: {}
      };

      for (const roi of rois) {
        const meanRgb = roi.rgb_summary?.mean_rgb;

        if (!meanRgb) {
          continue;
        }

        sample.rois[roi.name] = {
          r: meanRgb.r,
          g: meanRgb.g,
          b: meanRgb.b,
          quality_status: roi.quality?.status ?? "unknown"
        };
      }

      return sample;
    }

    function getRoiGreenSeries(roiName) {
      return roiSamples
        .filter(sample => sample.rois[roiName] !== undefined)
        .map(sample => {
          return {
            t: sample.t_s,
            g: sample.rois[roiName].g,
            quality_status: sample.rois[roiName].quality_status
          };
        });
    }

    function zscoreSeries(series) {
      if (series.length === 0) {
        return [];
      }

      const values = series.map(point => point.g);
      const seriesMean = mean(values);

      const variance =
        values.reduce((sum, value) => sum + Math.pow(value - seriesMean, 2), 0) /
        values.length;

      const std = Math.sqrt(variance);

      if (std < 1e-8) {
        return series.map(point => {
          return {
            t: point.t,
            z: 0.0,
            quality_status: point.quality_status
          };
        });
      }

      return series.map(point => {
        return {
          t: point.t,
          z: (point.g - seriesMean) / std,
          quality_status: point.quality_status
        };
      });
    }

    function drawTracePlot(options) {
      const canvasEl = document.getElementById(options.canvasId);

      if (!canvasEl) {
        return;
      }

      const context = canvasEl.getContext("2d");
      const width = canvasEl.width;
      const height = canvasEl.height;

      context.clearRect(0, 0, width, height);

      context.fillStyle = "white";
      context.fillRect(0, 0, width, height);

      const paddingLeft = 58;
      const paddingRight = 20;
      const paddingTop = 28;
      const paddingBottom = 38;

      const plotX0 = paddingLeft;
      const plotX1 = width - paddingRight;
      const plotY0 = paddingTop;
      const plotY1 = height - paddingBottom;

      context.strokeStyle = "#cbd5e1";
      context.lineWidth = 1;

      context.beginPath();
      context.moveTo(plotX0, plotY1);
      context.lineTo(plotX1, plotY1);
      context.moveTo(plotX0, plotY0);
      context.lineTo(plotX0, plotY1);
      context.stroke();

      context.fillStyle = "#0f172a";
      context.font = "14px sans-serif";
      context.fillText(options.title, plotX0, 18);

      if (roiSamples.length < 2) {
        context.fillStyle = "#64748b";
        context.font = "13px sans-serif";
        context.fillText("Collect ROI samples to plot traces.", plotX0, 60);
        return;
      }

      const plottedSeries = {};
      let allYValues = [];

      for (const roiName of ROI_NAMES) {
        const rawSeries = getRoiGreenSeries(roiName);

        if (options.mode === "zscore") {
          plottedSeries[roiName] = zscoreSeries(rawSeries);
          allYValues = allYValues.concat(
            plottedSeries[roiName].map(point => point.z)
          );
        } else {
          plottedSeries[roiName] = rawSeries;
          allYValues = allYValues.concat(
            plottedSeries[roiName].map(point => point.g)
          );
        }
      }

      if (allYValues.length === 0) {
        return;
      }

      const tMin = roiSamples[0].t_s;
      const tMax = roiSamples[roiSamples.length - 1].t_s;
      const yMinRaw = Math.min(...allYValues);
      const yMaxRaw = Math.max(...allYValues);

      const yPad = Math.max(0.25, (yMaxRaw - yMinRaw) * 0.20);
      const yMin = yMinRaw - yPad;
      const yMax = yMaxRaw + yPad;

      function xScale(t) {
        if (tMax <= tMin) {
          return plotX0;
        }

        return plotX0 + (t - tMin) / (tMax - tMin) * (plotX1 - plotX0);
      }

      function yScale(yValue) {
        if (yMax <= yMin) {
          return (plotY0 + plotY1) / 2;
        }

        return plotY1 - (yValue - yMin) / (yMax - yMin) * (plotY1 - plotY0);
      }

      context.strokeStyle = "#e2e8f0";
      context.lineWidth = 1;

      for (let i = 1; i <= 3; i += 1) {
        const y = plotY0 + i / 4 * (plotY1 - plotY0);
        context.beginPath();
        context.moveTo(plotX0, y);
        context.lineTo(plotX1, y);
        context.stroke();
      }

      if (options.mode === "zscore") {
        const zeroY = yScale(0.0);

        context.strokeStyle = "#94a3b8";
        context.setLineDash([5, 5]);
        context.beginPath();
        context.moveTo(plotX0, zeroY);
        context.lineTo(plotX1, zeroY);
        context.stroke();
        context.setLineDash([]);
      }

      for (const roiName of ROI_NAMES) {
        const series = plottedSeries[roiName];

        if (series.length < 2) {
          continue;
        }

        context.strokeStyle = getRoiColor(roiName);
        context.lineWidth = 2;
        context.beginPath();

        for (let i = 0; i < series.length; i += 1) {
          const yValue = options.mode === "zscore" ? series[i].z : series[i].g;
          const x = xScale(series[i].t);
          const y = yScale(yValue);

          if (i === 0) {
            context.moveTo(x, y);
          } else {
            context.lineTo(x, y);
          }
        }

        context.stroke();
      }

      context.fillStyle = "#64748b";
      context.font = "11px sans-serif";
      context.fillText(`${tMin.toFixed(1)}s`, plotX0, height - 14);
      context.fillText(`${tMax.toFixed(1)}s`, plotX1 - 34, height - 14);
      context.fillText(`${options.yLabel} min ${yMinRaw.toFixed(2)}`, 8, plotY1);
      context.fillText(`${options.yLabel} max ${yMaxRaw.toFixed(2)}`, 8, plotY0 + 8);

      let legendX = plotX0 + 260;
      const legendY = 18;

      for (const roiName of ROI_NAMES) {
        context.fillStyle = getRoiColor(roiName);
        context.fillRect(legendX, legendY - 9, 10, 10);

        context.fillStyle = "#0f172a";
        context.font = "11px sans-serif";
        context.fillText(roiName, legendX + 14, legendY);

        legendX += 150;
      }
    }

    function drawRoiGreenTracePlot() {
      drawTracePlot({
        canvasId: "roi-green-trace-canvas",
        title: "Raw ROI green-channel traces",
        mode: "raw",
        yLabel: "G"
      });
    }

    function drawNormalizedRoiGreenTracePlot() {
      drawTracePlot({
        canvasId: "roi-green-normalized-trace-canvas",
        title: "Normalized ROI green-channel traces",
        mode: "zscore",
        yLabel: "z"
      });
    }

    function drawAllRoiPlots() {
      drawRoiGreenTracePlot();
      drawNormalizedRoiGreenTracePlot();
    }

    
    function smoothSeriesForDisplay(values, windowSize = 5) {
  if (!Array.isArray(values) || values.length === 0) {
    return [];
  }

  if (windowSize <= 1 || values.length < windowSize) {
    return values.slice();
  }

  const halfWindow = Math.floor(windowSize / 2);
  const smoothed = [];

  for (let i = 0; i < values.length; i += 1) {
    let sum = 0.0;
    let count = 0;

    for (let j = i - halfWindow; j <= i + halfWindow; j += 1) {
      if (j < 0 || j >= values.length) {
        continue;
      }

      sum += values[j];
      count += 1;
    }

    smoothed.push(sum / count);
  }

  return smoothed;
}

function drawMonitorStylePulseWaveform(options) {
  const canvasEl = document.getElementById("main-pulse-wave-canvas");

  if (!canvasEl) {
    return;
  }

  const context = canvasEl.getContext("2d");
  const width = canvasEl.width;
  const height = canvasEl.height;

  context.clearRect(0, 0, width, height);

  context.fillStyle = "white";
  context.fillRect(0, 0, width, height);

  const paddingLeft = 28;
  const paddingRight = 28;
  const paddingTop = 18;
  const paddingBottom = 18;

  const plotX0 = paddingLeft;
  const plotX1 = width - paddingRight;
  const plotY0 = paddingTop;
  const plotY1 = height - paddingBottom;

  const values = Array.isArray(options.values) ? options.values : [];
  const times = Array.isArray(options.times) ? options.times : [];

  if (values.length < 2) {
    context.fillStyle = "#94a3b8";
    context.font = "18px sans-serif";
    context.fillText(
      options.emptyMessage ?? "Start sampling to display pulse waveform.",
      plotX0 + 10,
      (plotY0 + plotY1) / 2
    );
    return;
  }

  const displayValues = smoothSeriesForDisplay(values, options.smoothingWindow ?? 5);

  const yMinRaw = Math.min(...displayValues);
  const yMaxRaw = Math.max(...displayValues);

  const yPad = Math.max(0.35, (yMaxRaw - yMinRaw) * 0.28);
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;

  const hasTimes = times.length === displayValues.length;
  const xValues = hasTimes ? times : displayValues.map((_, index) => index);

  const tMin = xValues[0];
  const tMax = xValues[xValues.length - 1];

  function xScale(t) {
    if (tMax <= tMin) {
      return plotX0;
    }

    return plotX0 + (t - tMin) / (tMax - tMin) * (plotX1 - plotX0);
  }

  function yScale(value) {
    if (yMax <= yMin) {
      return (plotY0 + plotY1) / 2;
    }

    return plotY1 - (value - yMin) / (yMax - yMin) * (plotY1 - plotY0);
  }

  context.strokeStyle = "#21b8bd";
  context.lineWidth = 2.25;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();

  for (let i = 0; i < displayValues.length; i += 1) {
    const x = xScale(xValues[i]);
    const y = yScale(displayValues[i]);

    if (i === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }

  context.stroke();

  if (options.footerText) {
    context.fillStyle = "#64748b";
    context.font = "12px sans-serif";
    context.fillText(options.footerText, plotX0, height - 6);
  }
}

function drawMainPulseWaveformPlaceholder() {
  drawMonitorStylePulseWaveform({
    values: [],
    times: [],
    emptyMessage: "Start ROI sampling to display pulse waveform."
  });
}

function drawMainPulseWaveformFromLiveSamples() {
  if (roiSamples.length < 2) {
    drawMainPulseWaveformPlaceholder();
    return;
  }

  const points = [];

  for (const sample of roiSamples) {
    const greenValues = [];

    for (const roiName of ROI_NAMES) {
      const roi = sample.rois?.[roiName];

      if (!roi) {
        continue;
      }

      if (roi.quality_status !== "ok" && roi.quality_status !== "warning") {
        continue;
      }

      const greenValue = Number(roi.g);

      if (!Number.isNaN(greenValue)) {
        greenValues.push(greenValue);
      }
    }

    if (greenValues.length === 0) {
      continue;
    }

    points.push({
      t: Number(sample.t_s),
      value: mean(greenValues)
    });
  }

  if (points.length < 2) {
    drawMonitorStylePulseWaveform({
      values: [],
      times: [],
      emptyMessage: "Waiting for usable ROI samples."
    });
    return;
  }

  const warmupSeconds = 3.0;
    const maxDisplaySeconds = 12.0;

    const latestT = points[points.length - 1].t;
    const minDisplayT = Math.max(warmupSeconds, latestT - maxDisplaySeconds);

    const visiblePoints = points.filter(point => point.t >= minDisplayT);

    if (latestT < warmupSeconds || visiblePoints.length < 8) {
    drawMonitorStylePulseWaveform({
        values: [],
        times: [],
        emptyMessage: "Stabilizing rPPG signal..."
    });
    return;
    }

  const rawValues = visiblePoints.map(point => point.value);
  const valueMean = mean(rawValues);

  const variance =
    rawValues.reduce((sum, value) => sum + Math.pow(value - valueMean, 2), 0) /
    rawValues.length;

  const std = Math.sqrt(variance);

  const normalizedValues = rawValues.map(value => {
    if (std < 1e-8) {
      return 0.0;
    }

    return (value - valueMean) / std;
  });

  const times = visiblePoints.map(point => point.t);

  drawMonitorStylePulseWaveform({
    values: normalizedValues,
    times: times,
    smoothingWindow: 7,
    footerText: `Live averaged ROI GREEN | samples ${normalizedValues.length}`
  });
}

function chooseBestPulseWaveformSignal(data) {
  const candidates = ["green", "pos", "chrom"];
  let bestSignal = null;

  for (const signalName of candidates) {
    const signalData = data.signals?.[signalName];
    const values = signalData?.values;
    const spectral = signalData?.spectral;

    if (!Array.isArray(values) || values.length < 2) {
      continue;
    }

    const sqi = getValidNumber(spectral?.sqi);
    const bpm = getValidNumber(spectral?.dominant_bpm);
    const status = spectral?.status ?? "unknown";

    if (sqi === null) {
      continue;
    }

    if (bestSignal === null || sqi > bestSignal.sqi) {
      bestSignal = {
        name: signalName,
        values: values.map(value => Number(value)),
        sqi: sqi,
        bpm: bpm,
        status: status
      };
    }
  }

  return bestSignal;
}

function drawMainPulseWaveformFromAnalysis(data) {
  const selectedSignal = chooseBestPulseWaveformSignal(data);

  if (selectedSignal === null) {
    drawMonitorStylePulseWaveform({
      values: [],
      times: [],
      emptyMessage: "No usable rPPG waveform returned by backend analysis."
    });
    return;
  }

  const timeValues =
    Array.isArray(data.time_s) && data.time_s.length === selectedSignal.values.length
      ? data.time_s.map(value => Number(value))
      : selectedSignal.values.map((_, index) => index);

  const sourceLabel = selectedSignal.name.toUpperCase();

  const bpmText =
    selectedSignal.bpm === null
      ? "HR none"
      : `${selectedSignal.bpm.toFixed(1)} bpm`;

  const footerText =
    `Source: ${sourceLabel} | ${bpmText} | SQI ${selectedSignal.sqi.toFixed(3)} | ${selectedSignal.status}`;

  drawMonitorStylePulseWaveform({
    values: selectedSignal.values,
    times: timeValues,
    smoothingWindow: 5,
    footerText: footerText
  });
}
     

  function summarizeCollectedRoiSamples() {
      const summaryEl = document.getElementById("roi-sampling-summary");

      if (roiSamples.length === 0) {
        if (summaryEl) {
          summaryEl.innerText = "No ROI samples collected yet.";
        }

        drawAllRoiPlots();
        drawMainPulseWaveformPlaceholder();
        updatePrimaryControlButtons();
        return;
      }

      const firstT = roiSamples[0].t_s;
      const lastT = roiSamples[roiSamples.length - 1].t_s;
      const durationS = Math.max(0, lastT - firstT);

      const faceDetectedCount = roiSamples
        .filter(sample => sample.face_detected)
        .length;

      const roiUsableSamples = roiSamples.filter(sample =>
        ROI_NAMES.some(roiName => sample.rois[roiName] !== undefined)
      );

      const roiUsableDurationS =
        roiUsableSamples.length > 1
          ? Math.max(
              0,
              roiUsableSamples[roiUsableSamples.length - 1].t_s - roiUsableSamples[0].t_s
            )
          : 0;

      const roiUsableEffectiveFps =
        roiUsableSamples.length > 1 && roiUsableDurationS > 0
          ? (roiUsableSamples.length - 1) / roiUsableDurationS
          : null;

      const timingEntries = roiSamples
        .map(sample => sample.acquisition_timing)
        .filter(value => value !== undefined && value !== null);

      const lines = [];

      lines.push(`samples: ${roiSamples.length}`);
      lines.push(`duration_s: ${durationS.toFixed(2)}`);
      lines.push(`face_detected_samples: ${faceDetectedCount}/${roiSamples.length}`);
      lines.push(`roi_usable_samples: ${roiUsableSamples.length}/${roiSamples.length}`);
      lines.push(`roi_usable_effective_fps: ${formatNumber(roiUsableEffectiveFps, 2)}`);

      if (timingEntries.length > 0) {
        lines.push(
          "timing_ms_mean: " +
          `capture=${formatNumber(meanFinite(timingEntries.map(value => value.capture_to_data_url_ms)), 1)}, ` +
          `round_trip=${formatNumber(meanFinite(timingEntries.map(value => value.request_round_trip_ms)), 1)}, ` +
          `backend=${formatNumber(meanFinite(timingEntries.map(value => value.backend_total_request_processing_ms)), 1)}, ` +
          `browser_total=${formatNumber(meanFinite(timingEntries.map(value => value.total_browser_sample_ms)), 1)}`
        );
      }

      lines.push("");

      for (const roiName of ROI_NAMES) {
        const values = roiSamples
          .map(sample => sample.rois[roiName])
          .filter(value => value !== undefined);

        if (values.length === 0) {
          lines.push(`${roiName}: no samples`);
          continue;
        }

        const latest = values[values.length - 1];
        const qualityCounts = {};

        for (const value of values) {
          qualityCounts[value.quality_status] =
            (qualityCounts[value.quality_status] ?? 0) + 1;
        }

        const greenValues = values.map(value => value.g);
        const greenMin = Math.min(...greenValues);
        const greenMax = Math.max(...greenValues);
        const greenRange = greenMax - greenMin;

        lines.push(
          `${roiName}: n=${values.length}, ` +
          `latest RGB=(${latest.r.toFixed(1)}, ${latest.g.toFixed(1)}, ${latest.b.toFixed(1)}), ` +
          `green_range=${greenRange.toFixed(2)}, ` +
          `quality_counts=${JSON.stringify(qualityCounts)}`
        );
      }

      lines.push("");
      lines.push("Note: this is raw ROI RGB only, not rPPG and not HR.");

      if (summaryEl) {
        summaryEl.innerText = lines.join("\n");
      }

      drawAllRoiPlots();
      drawMainPulseWaveformFromLiveSamples();
      updatePrimaryControlButtons();
    }


    function clearRoiSamplingSchedule() {
      if (roiSamplingScheduleId !== null) {
        clearTimeout(roiSamplingScheduleId);
        roiSamplingScheduleId = null;
      }
    }

    function scheduleNextRoiSample(expectedSamplingRunId, expectedRevision, delayMs = 0) {
      if (
        expectedSamplingRunId !== roiSamplingRunId ||
        expectedRevision !== measurementRevision ||
        roiSamplingScheduleId === null ||
        !cameraStream
      ) {
        return;
      }

      const safeDelayMs = Math.max(0, Number(delayMs) || 0);

      roiSamplingScheduleId = setTimeout(() => {
        collectOneRoiSample(expectedSamplingRunId, expectedRevision);
      }, safeDelayMs);
    }
  
    async function collectOneRoiSample(
      expectedSamplingRunId = roiSamplingRunId,
      expectedRevision = measurementRevision
    ) {
      const statusEl = document.getElementById("camera-status");
      let nextSampleDelayMs = 0;

      if (roiSamplingInFlight) {
        return;
      }

      roiSamplingInFlight = true;

      try {
        const sampleStartMs = performance.now();
        const captureStartMs = performance.now();
        const imageDataUrl = captureFrameForSamplingDataUrl();
        const captureEndMs = performance.now();

        if (!imageDataUrl) {
          nextSampleDelayMs = 100;
          return;
        }

        if (
          expectedSamplingRunId !== roiSamplingRunId ||
          expectedRevision !== measurementRevision
        ) {
          return;
        }

        const requestStartMs = performance.now();

        const response = await fetch("/api/roi-sample", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            image_data_url: imageDataUrl
          })
        });

        const responseReceivedMs = performance.now();
        const data = await parseJsonResponseEvenOnError(response);
        const responseParsedMs = performance.now();

        if (
          expectedSamplingRunId !== roiSamplingRunId ||
          expectedRevision !== measurementRevision ||
          roiSamplingScheduleId === null
        ) {
          return;
        }

        const sample = extractRoiSampleFromBackendResponse(data, {
          captureToDataUrlMs: captureEndMs - captureStartMs,
          requestRoundTripMs: responseReceivedMs - requestStartMs,
          responseParseMs: responseParsedMs - responseReceivedMs,
          totalBrowserSampleMs: responseParsedMs - sampleStartMs,
          httpStatus: response.status
        });

        roiSamples.push(sample);

        summarizeCollectedRoiSamples();

        if (statusEl) {
          statusEl.innerText =
            `ROI sampling active. Collected ${roiSamples.length} sample(s). ` +
            "Frames are processed in memory and not stored.";
        }
      } catch (error) {
        nextSampleDelayMs = 150;

        if (
          expectedSamplingRunId === roiSamplingRunId &&
          expectedRevision === measurementRevision &&
          roiSamplingScheduleId !== null &&
          statusEl
        ) {
          statusEl.innerText = `ROI sampling error: ${error}`;
        }
      } finally {
        if (expectedSamplingRunId === roiSamplingRunId) {
          roiSamplingInFlight = false;
        }

        scheduleNextRoiSample(
          expectedSamplingRunId,
          expectedRevision,
          nextSampleDelayMs
        );
      }
    }


    function startRoiSampling({ resetOutputs = true } = {}) {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      if (!cameraStream) {
        statusEl.innerText = "Cannot start ROI sampling: camera is not started.";
        updatePrimaryControlButtons();
        return false;
      }

      if (roiSamplingScheduleId !== null) {
        statusEl.innerText = "ROI sampling is already running.";
        updatePrimaryControlButtons();
        return false;
      }

      roiSamplingRunId += 1;
      const activeSamplingRunId = roiSamplingRunId;
      const activeRevision = measurementRevision;

      roiSamples = [];
      roiSamplingStartMs = performance.now();
      roiSamplingInFlight = false;

      if (resetOutputs) {
        resetMeasurementOutputs(activeRevision);
      }

      if (startButton) {
        startButton.disabled = true;
        startButton.innerText = "Sampling...";
      }

      statusEl.innerText =
        "ROI sampling started in completion-driven mode. Hold still while each backend ROI sample completes.";

      summarizeCollectedRoiSamples();

      roiSamplingScheduleId = 0;
      scheduleNextRoiSample(activeSamplingRunId, activeRevision, 0);
      updatePrimaryControlButtons();

      return true;
    }

    function stopRoiSampling() {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      roiSamplingRunId += 1;
      roiSamplingInFlight = false;
      clearRoiSamplingSchedule();

      if (startButton) {
        startButton.disabled = false;
        startButton.innerText = "Start ROI sampling";
      }

      summarizeCollectedRoiSamples();

      if (statusEl) {
        statusEl.innerText =
          `ROI sampling stopped. Collected ${roiSamples.length} sample(s).`;
      }

      updatePrimaryControlButtons();
    }

    function clearRoiSamples() {
      const statusEl = document.getElementById("camera-status");

      setMainMeasurementButtonsState(false);
      measurementRevision += 1;
      const activeRevision = measurementRevision;

      resetMeasurementRunState(activeRevision, {
        resetCapturedFrame: true,
      });

      if (cameraStream) {
        setMeasurementStatus("Ready.", "Start a measurement while holding still.");
      } else {
        setMeasurementStatus("Camera stopped.", "Start the camera before running a measurement.");
      }

      if (statusEl) {
        statusEl.innerText = "ROI samples, captured frame, and measurement results cleared.";
      }

      updatePrimaryControlButtons();
    }

    async function startMainMeasurement() {
      const statusEl = document.getElementById("camera-status");
      const totalSeconds = MAIN_MEASUREMENT_DURATION_MS / 1000.0;
      const mobileViewport = isMobileDemoViewport();

      if (mainMeasurementInProgress) {
        statusEl.innerText = "Measurement is already running.";
        return;
      }

      if (!cameraStream) {
        if (!mobileViewport) {
          statusEl.innerText = "Cannot start measurement: camera is not started.";
          return;
        }

        setMeasurementStatus(
          "Starting camera...",
          "Camera permission is needed before the measurement can begin."
        );

        if (statusEl) {
          statusEl.innerText = "Starting camera for mobile measurement...";
        }

        const cameraStarted = await startCameraPreview();

        if (!cameraStarted || !cameraStream) {
          setMeasurementStatus(
            "Camera unavailable.",
            "Camera permission or browser security blocked the measurement."
          );
          return;
        }
      }

      const videoReady = await waitForVideoPreviewReady();

      if (!videoReady) {
        if (statusEl) {
          statusEl.innerText = "Cannot start measurement: camera preview is not ready yet.";
        }

        setMeasurementStatus(
          "Measurement could not start.",
          "Camera preview is not ready yet. Try again in a moment."
        );
        return;
      }

      measurementRevision += 1;
      const activeRevision = measurementRevision;

      resetMeasurementRunState(activeRevision, {
        resetCapturedFrame: true,
      });

      mainMeasurementInProgress = true;
      setMainMeasurementButtonsState(true);

      if (mobileViewport) {
        const prepareCompleted = await runMobileMeasurementPrepareCountdown(activeRevision, 3);

        if (!prepareCompleted || activeRevision !== measurementRevision) {
          return;
        }
      }

      const measurementStartedAtMs = performance.now();

      if (mobileViewport) {
        setMeasurementStatus(
          "Measuring...",
          `Collecting rPPG signal. ${totalSeconds.toFixed(1)} s remaining.`
        );
        setMeasurementProgress(0.0, `${totalSeconds.toFixed(1)} s remaining`);
      } else {
        setMeasurementStatus(
          "Measuring...",
          `Stabilizing signal. 0.0 / ${totalSeconds.toFixed(1)} s`
        );
        setMeasurementProgress(0.0, `0.0 / ${totalSeconds.toFixed(1)} s`);
      }

      startMeasurementProgressTimer(activeRevision, measurementStartedAtMs);

      const samplingStarted = startRoiSampling({
        resetOutputs: false,
      });

      if (!samplingStarted || roiSamplingScheduleId === null) {
        mainMeasurementInProgress = false;
        stopMeasurementProgressTimer();
        setMainMeasurementButtonsState(false);
        hideMobileCameraGuidanceOverlay();
        setMeasurementStatus("Measurement could not start.", "Camera or video frame was not ready.");
        setMeasurementProgress(0.0, "0%");
        return;
      }

      statusEl.innerText =
        `Measurement started. Hold still for ${totalSeconds.toFixed(0)} seconds.`;

      mainMeasurementTimer = setTimeout(async () => {
        mainMeasurementTimer = null;
        stopMeasurementProgressTimer();

        try {
          if (activeRevision !== measurementRevision) {
            return;
          }

          setMeasurementProgress(
            1.0,
            mobileViewport
              ? "0.0 s remaining"
              : `${totalSeconds.toFixed(1)} / ${totalSeconds.toFixed(1)} s`
          );
          setMeasurementStatus("Measurement complete.", "Running backend rPPG analysis...");

          stopRoiSampling();

          if (activeRevision !== measurementRevision) {
            return;
          }

          statusEl.innerText =
            `Measurement complete. Collected ${roiSamples.length} sample(s). Running backend analysis...`;
          setMeasurementStatus(
            "Analyzing signal...",
            `Collected ${roiSamples.length} sample(s). Computing spectral HR.`
          );

          await analyzeRoiSeriesInBackend(activeRevision);

          if (activeRevision !== measurementRevision) {
            return;
          }

          statusEl.innerText = "Analysis complete. Running experimental model prediction...";
          setMeasurementStatus(
            "Running model prediction...",
            "Computing experimental CRVSE PhysFormer HR estimate."
          );

          const predictionStatus = await runLiveModelPredictionInBackend(activeRevision);

          if (activeRevision === measurementRevision && predictionStatus !== null) {
            setMeasurementStatus(
              predictionStatus.summary,
              predictionStatus.detail
            );
          }
        } catch (error) {
          if (activeRevision === measurementRevision) {
            statusEl.innerText = `Measurement flow failed: ${error}`;
            setMeasurementStatus("Measurement failed.", `${error}`);
          }
        } finally {
          if (activeRevision === measurementRevision) {
            mainMeasurementInProgress = false;
            stopMeasurementProgressTimer();
            setMainMeasurementButtonsState(false);
            hideMobileCameraGuidanceOverlay();
            scrollToMeasurementResultsOnMobile();
          }
        }
      }, MAIN_MEASUREMENT_DURATION_MS);
    }

    function stopMainMeasurement() {
      const videoEl = document.getElementById("camera-video");
      const statusEl = document.getElementById("camera-status");

      if (mainMeasurementTimer !== null) {
        clearTimeout(mainMeasurementTimer);
        mainMeasurementTimer = null;
      }

      measurementRevision += 1;
      const activeRevision = measurementRevision;

      stopMeasurementProgressTimer();

      if (isMobileDemoViewport()) {
        if (cameraStream) {
          const tracks = cameraStream.getTracks();

          for (const track of tracks) {
            track.stop();
          }

          cameraStream = null;
        }

        if (videoEl) {
          videoEl.srcObject = null;
        }

        resetMeasurementRunState(activeRevision, {
          resetCapturedFrame: true,
        });

        setMainMeasurementButtonsState(false);
        setMeasurementStatus(
          "Measurement stopped.",
          "Mobile measurement stopped, camera closed, and outputs cleared."
        );
        setMeasurementProgress(0.0, "Stopped");

        if (statusEl) {
          statusEl.innerText =
            "Measurement stopped. Camera closed and measurement output cleared.";
        }

        updatePrimaryControlButtons();
        return;
      }

      if (roiSamplingScheduleId !== null) {
        stopRoiSampling();
      }

      setMainMeasurementButtonsState(false);

      statusEl.innerText =
        `Measurement stopped. Collected ${roiSamples.length} sample(s). Camera is still active.`;
      setMeasurementStatus(
        "Measurement stopped.",
        `Collected ${roiSamples.length} sample(s). You can analyze manually from diagnostics or clear and retry.`
      );
      setMeasurementProgress(0.0, "Stopped");

      updatePrimaryControlButtons();
    }

    function summarizeSpectralQualityFromEntries(entries, contextLabel = "Spectral gate") {
      const validEntries = entries.filter(entry => entry !== null);

      if (validEntries.length === 0) {
        return {
          summary: "Not available",
          detail: `${contextLabel}: no spectral channel results returned.`
        };
      }

      const goodEntries = validEntries.filter(entry => entry.status === "good");
      const moderateEntries = validEntries.filter(entry => entry.status === "moderate");
      const supportedEntries = goodEntries.concat(moderateEntries);

      const supportedBpmValues = supportedEntries
        .map(entry => getValidNumber(entry.bpm))
        .filter(value => value !== null);

      const minSupportedChannels = 2;
      const maxAllowedSpread = 20.0;

      if (supportedEntries.length < minSupportedChannels) {
        return {
          summary: "Rejected",
          detail: `${contextLabel}: fewer than ${minSupportedChannels} channels have moderate-or-better spectral support.`
        };
      }

      if (supportedBpmValues.length < minSupportedChannels) {
        return {
          summary: "Rejected",
          detail: `${contextLabel}: not enough supported channels have valid dominant HR peaks.`
        };
      }

      const bpmMin = Math.min(...supportedBpmValues);
      const bpmMax = Math.max(...supportedBpmValues);
      const bpmSpread = bpmMax - bpmMin;

      if (bpmSpread > maxAllowedSpread) {
        return {
          summary: "Rejected",
          detail: `${contextLabel}: supported-channel HR peaks disagree, spread ${bpmSpread.toFixed(1)} bpm.`
        };
      }

      if (goodEntries.length > 0) {
        return {
          summary: "Accepted / good",
          detail: `${contextLabel}: ${supportedEntries.length} supported channel(s), ${goodEntries.length} good, spread ${bpmSpread.toFixed(1)} bpm.`
        };
      }

      return {
        summary: "Accepted / moderate",
        detail: `${contextLabel}: ${supportedEntries.length} moderate channel(s), spread ${bpmSpread.toFixed(1)} bpm.`
      };
    }

    function updateMeasurementQualityFromAnalysis(data) {
      const entries = ["green", "pos", "chrom"].map(signalName => {
        const spectral = data.signals?.[signalName]?.spectral;

        if (!spectral) {
          return null;
        }

        return {
          name: signalName,
          bpm: spectral.dominant_bpm,
          sqi: spectral.sqi,
          status: spectral.status ?? "unknown"
        };
      });

      return summarizeSpectralQualityFromEntries(entries, "Full-buffer spectral gate");
    }

    function updateMeasurementQualityFromModelPrediction(data) {
      const summary = data.classical_spectral_summary ?? {};

      const entries = ["green", "pos", "chrom"].map(signalName => {
        const spectral = summary[signalName];

        if (!spectral) {
          return null;
        }

        return {
          name: signalName,
          bpm: spectral.dominant_bpm,
          sqi: spectral.sqi,
          status: spectral.status ?? "unknown"
        };
      });

      return summarizeSpectralQualityFromEntries(entries, "Model-window spectral gate");
    }

    function computeSpectralConsensusFromAnalysis(data) {
      const values = [];

      for (const signalName of ["green", "pos", "chrom"]) {
        const bpm = getValidNumber(data.signals?.[signalName]?.spectral?.dominant_bpm);

        if (bpm !== null) {
          values.push(bpm);
        }
      }

      return mean(values);
    }

    
    async function analyzeRoiSeriesInBackend(expectedRevision = null) {
      if (typeof expectedRevision !== "number") {
        expectedRevision = null;
      }

      const statusEl = document.getElementById("camera-status");
      const analyzeButton = document.getElementById("analyze-roi-series-button");

      function formatSignalSummary(signalName, signalData) {
        const spectral = signalData?.spectral;

        if (!spectral) {
          return `${signalName}: unavailable`;
        }

        const bpm = spectral.dominant_bpm;
        const sqi = spectral.sqi;
        const spectralStatus = spectral.status ?? "unknown";

        const bpmText =
          bpm === null || bpm === undefined
            ? "no BPM"
            : `${bpm.toFixed(1)} bpm`;

        const sqiText =
          sqi === null || sqi === undefined
            ? "no SQI"
            : `SQI ${sqi.toFixed(3)}`;

        return `${bpmText} / ${sqiText} / ${spectralStatus}`;
      }

      function formatSignalDetail(signalName, signalData) {
        const spectral = signalData?.spectral;

        if (!spectral) {
          return `${signalName} signal unavailable`;
        }

        const dominantHz = getValidNumber(spectral.dominant_hz);
        const status = spectral.status ?? "unknown";

        if (dominantHz === null) {
          return `${signalName} spectral status: ${status}`;
        }

        return `${signalName} dominant frequency ${dominantHz.toFixed(3)} Hz`;
      }

      if (roiSamples.length < 20) {
        statusEl.innerText = "Collect at least 20 ROI samples before analysis.";
        return null;
      }

      if (analyzeButton) {
        analyzeButton.disabled = true;
        analyzeButton.innerText = "Analyzing...";
      }

      try {
        const response = await fetch("/api/analyze-roi-series", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            samples: roiSamples
          })
        });

        const data = await parseJsonResponseEvenOnError(response);

        if (expectedRevision !== null && expectedRevision !== measurementRevision) {
          return null;
        }

        drawMainPulseWaveformFromAnalysis(data);

        const greenBpm = data.signals?.green?.spectral?.dominant_bpm;
        const posBpm = data.signals?.pos?.spectral?.dominant_bpm;
        const chromBpm = data.signals?.chrom?.spectral?.dominant_bpm;

        const spectralConsensus = computeSpectralConsensusFromAnalysis(data);

        const quality =
          updateMeasurementQualityFromAnalysis(data) ?? {
            summary: "Not available",
            detail: "Signal quality summary was not returned."
          };

        const diagnosticSpectralHrText = formatBpm(spectralConsensus);
        const fullBufferRejected = quality.summary === "Rejected";

        const primarySpectralHrText = fullBufferRejected
          ? "Rejected"
          : diagnosticSpectralHrText;

        const primarySpectralDetail = fullBufferRejected
          ? (
              `Full-buffer spectral candidate was ${diagnosticSpectralHrText}, ` +
              "but the quality gate rejected this window. Review channel SQI and diagnostics."
            )
          : "Primary estimate: full-buffer spectral consensus from GREEN / POS / CHROM";

        latestRoiAnalysisDisplayState = {
          revision: measurementRevision,
          sampleCount: roiSamples.length,
          spectralBpm: spectralConsensus,
          spectralHr: primarySpectralHrText,
          spectralDetail: primarySpectralDetail,
          diagnosticSpectralHr: diagnosticSpectralHrText,
          quality: quality.summary,
          qualityDetail: quality.detail,
        };

        await renderSignalSummaryCardsFromServer({
          green: formatSignalSummary("GREEN", data.signals?.green),
          greenDetail: formatSignalDetail("GREEN", data.signals?.green),
          pos: formatSignalSummary("POS", data.signals?.pos),
          posDetail: formatSignalDetail("POS", data.signals?.pos),
          chrom: formatSignalSummary("CHROM", data.signals?.chrom),
          chromDetail: formatSignalDetail("CHROM", data.signals?.chrom),
        });

        await renderMeasurementResultCardsFromServer({
          spectralHr: primarySpectralHrText,
          spectralDetail: primarySpectralDetail,
          modelHr: "Not predicted yet",
          modelDetail: "Run live model prediction to compare with model-window spectral HR",
          modelDifference: "Not predicted yet",
          modelDifferenceDetail: "Model-vs-model-window spectral agreement diagnostic",
          quality: quality.summary,
          qualityDetail: quality.detail,
        });

        await renderRoiAnalysisSummaryFromServer({
          status: data.status ?? "unknown",
          sampleCount: String(data.sample_count ?? "none"),
          durationS: `${formatNumber(data.duration_s, 2)} s`,
          estimatedFps: `${formatNumber(data.estimated_fps, 2)} Hz`,
          spectralConsensus: diagnosticSpectralHrText,
          greenSummary: formatSignalSummary("GREEN", data.signals?.green),
          posSummary: formatSignalSummary("POS", data.signals?.pos),
          chromSummary: formatSignalSummary("CHROM", data.signals?.chrom),
          rawResponse: JSON.stringify(buildCompactRoiAnalysisDebugResponse(data), null, 2),
        });

        if (fullBufferRejected) {
          statusEl.innerText =
            `ROI series analyzed but rejected by full-buffer spectral gate. ` +
            `Candidate consensus=${diagnosticSpectralHrText}. ` +
            `GREEN=${greenBpm?.toFixed(1) ?? "none"} BPM, ` +
            `POS=${posBpm?.toFixed(1) ?? "none"} BPM, ` +
            `CHROM=${chromBpm?.toFixed(1) ?? "none"} BPM.`;

          return latestRoiAnalysisDisplayState;
        }

        statusEl.innerText =
          `ROI series analyzed. Full-buffer spectral consensus=${diagnosticSpectralHrText}. ` +
          `GREEN=${greenBpm?.toFixed(1) ?? "none"} BPM, ` +
          `POS=${posBpm?.toFixed(1) ?? "none"} BPM, ` +
          `CHROM=${chromBpm?.toFixed(1) ?? "none"} BPM.`;

        return latestRoiAnalysisDisplayState;
      } catch (error) {
        if (expectedRevision === null || expectedRevision === measurementRevision) {
          latestRoiAnalysisDisplayState = null;
        }

        statusEl.innerText = `ROI series analysis failed: ${error}`;

        drawMainPulseWaveformPlaceholder();

        await renderSignalSummaryCardsFromServer({
          green: "Analysis failed",
          greenDetail: "GREEN signal could not be analyzed",
          pos: "Analysis failed",
          posDetail: "POS signal could not be analyzed",
          chrom: "Analysis failed",
          chromDetail: "CHROM signal could not be analyzed",
        });

        await renderMeasurementResultCardsFromServer({
          spectralHr: "Analysis failed",
          spectralDetail: "Full-buffer spectral HR could not be computed",
          modelHr: "Not predicted yet",
          modelDetail: "Model prediction was not run",
          modelDifference: "Not predicted yet",
          modelDifferenceDetail: "Agreement diagnostic unavailable",
          quality: "Analysis failed",
          qualityDetail: "Full-buffer spectral quality could not be computed.",
        });

        await renderRoiAnalysisSummaryFromServer({
          status: "Analysis failed",
          sampleCount: String(roiSamples.length),
          durationS: "none",
          estimatedFps: "none",
          spectralConsensus: "none",
          greenSummary: "GREEN signal could not be analyzed",
          posSummary: "POS signal could not be analyzed",
          chromSummary: "CHROM signal could not be analyzed",
          rawResponse: `${error}`,
        });

        return null;
      } finally {
        if (analyzeButton) {
          analyzeButton.disabled = false;
          analyzeButton.innerText = "Analyze ROI series";
        }
      }
    }

    function spectralBpmValues(data) {
      const greenBpm = getValidNumber(data.classical_spectral_summary?.green?.dominant_bpm);
      const posBpm = getValidNumber(data.classical_spectral_summary?.pos?.dominant_bpm);
      const chromBpm = getValidNumber(data.classical_spectral_summary?.chrom?.dominant_bpm);

      return [greenBpm, posBpm, chromBpm].filter(value => value !== null);
    }

    function isModelUnavailableResponse(data) {
      return data?.model_available === false || data?.status === "model_unavailable";
    }

    function modelUnavailableDetail(data) {
      return (
        data?.message ??
        "Experimental model prediction is unavailable. Spectral rPPG analysis remains available."
      );
    }

    function formatSecondsOrNone(value) {
      const formatted = formatNumber(value, 2);
      return formatted === "none" ? "none" : `${formatted} s`;
    }

    function formatHzOrNone(value) {
      const formatted = formatNumber(value, 2);
      return formatted === "none" ? "none" : `${formatted} Hz`;
    }

    function modelSpectralAgreementDetail(difference) {
      /*
      Return cautious wording for model-vs-model-window-spectral agreement.
      */

      const value = getValidNumber(difference);

      if (value === null) {
        return "Agreement diagnostic unavailable.";
      }

      const absValue = Math.abs(value);

      if (absValue >= 10) {
        return "Model differs from model-window spectral consensus; treat model HR as experimental.";
      }

      if (absValue >= 5) {
        return "Model and model-window spectral estimates differ moderately.";
      }

      return "Model and model-window spectral estimates are close in this window.";
    }

    function summarizeModelPrediction(data) {
      const modelHr = getValidNumber(data.model_prediction?.value);
      const qualityStatus = data.model_prediction?.quality?.status ?? null;
      const gatedSpectralHr = getValidNumber(data.model_prediction?.extra?.spectral_hr_bpm);
      const rawSpectralConsensus = mean(spectralBpmValues(data));

      let consensus = gatedSpectralHr;

      if (
        consensus === null &&
        (isModelUnavailableResponse(data) || qualityStatus === "accepted")
      ) {
        consensus = rawSpectralConsensus;
      }

      const difference =
        modelHr !== null && consensus !== null
          ? modelHr - consensus
          : null;

      const green = data.classical_spectral_summary?.green;
      const pos = data.classical_spectral_summary?.pos;
      const chrom = data.classical_spectral_summary?.chrom;
      const windowMetadata = data.model_input?.window_metadata ?? {};

      return {
        modelHr,
        consensus,
        difference,
        qualityStatus,
        rawSpectralConsensus,
        greenBpm: getValidNumber(green?.dominant_bpm),
        posBpm: getValidNumber(pos?.dominant_bpm),
        chromBpm: getValidNumber(chrom?.dominant_bpm),
        greenSqi: getValidNumber(green?.sqi),
        posSqi: getValidNumber(pos?.sqi),
        chromSqi: getValidNumber(chrom?.sqi),
        greenStatus: green?.status ?? "unknown",
        posStatus: pos?.status ?? "unknown",
        chromStatus: chrom?.status ?? "unknown",
        originalDurationS: getValidNumber(windowMetadata.original_duration_s),
        usedDurationS: getValidNumber(windowMetadata.used_duration_s),
        usedSamples: getValidNumber(windowMetadata.used_sample_count),
        sourceFps: getValidNumber(data.model_input?.source_estimated_fps),
      };
    }

    function buildRepeatabilityRunsForServer() {
      if (!window.livePredictionRuns) {
        return [];
      }

      return window.livePredictionRuns.map(run => {
        return {
          run_index: String(run.runIndex ?? ""),
          model_hr: formatBpm(run.modelHr),
          spectral: formatBpm(run.consensus),
          difference: formatSignedBpm(run.difference),
          green_sqi: `${formatNumber(run.greenSqi, 3)} / ${run.greenStatus ?? "unknown"}`,
          pos_sqi: `${formatNumber(run.posSqi, 3)} / ${run.posStatus ?? "unknown"}`,
          chrom_sqi: `${formatNumber(run.chromSqi, 3)} / ${run.chromStatus ?? "unknown"}`,
          used_seconds: formatNumber(run.usedDurationS, 2),
          samples: run.usedSamples ?? "none",
          fps: formatNumber(run.sourceFps, 2)
        };
      });
    }

    async function renderRepeatabilityTableFromServer() {
      /*
      Fetch a server-rendered repeatability table and insert it below the
      live model prediction details block.

      Camera, sampling, and waveform drawing remain browser-side. The table
      layout is rendered by FastHTML / MonsterUI components.
      */

      const panelContainer = document.getElementById("live-model-prediction-output-container");

      if (!panelContainer) {
        return;
      }

      const html = await fetchServerPartialHtml(
        "/ui/repeatability-table-json",
        {
          runs: buildRepeatabilityRunsForServer(),
        },
        "POST"
      );
      const existingContainer = document.getElementById("live-model-repeatability-container");

      if (existingContainer) {
        existingContainer.outerHTML = html;
        return;
      }

      panelContainer.insertAdjacentHTML("afterend", html);
    }

    async function addPredictionRun(summary) {
      const runIndex = window.livePredictionRuns.length + 1;

      window.livePredictionRuns.push({
        runIndex: runIndex,
        modelHr: summary.modelHr,
        consensus: summary.consensus,
        difference: summary.difference,
        greenBpm: summary.greenBpm,
        posBpm: summary.posBpm,
        chromBpm: summary.chromBpm,
        greenSqi: summary.greenSqi,
        posSqi: summary.posSqi,
        chromSqi: summary.chromSqi,
        greenStatus: summary.greenStatus,
        posStatus: summary.posStatus,
        chromStatus: summary.chromStatus,
        originalDurationS: summary.originalDurationS,
        usedDurationS: summary.usedDurationS,
        usedSamples: summary.usedSamples,
        sourceFps: summary.sourceFps
      });

      await renderRepeatabilityTableFromServer();
    }


    function estimateRoiSamplesEffectiveFpsForModel(samples, windowSeconds = 12.0) {
      const validSamples = samples
        .filter(sample => {
          const t = Number(sample?.t_s);
          const hasAllRois = ROI_NAMES.every(roiName => sample?.rois?.[roiName] !== undefined);
          return Number.isFinite(t) && hasAllRois;
        })
        .map(sample => ({
          ...sample,
          t_s: Number(sample.t_s)
        }))
        .sort((a, b) => a.t_s - b.t_s);

      if (validSamples.length < 2) {
        return null;
      }

      const latestTimeS = validSamples[validSamples.length - 1].t_s;
      const cutoffS = latestTimeS - Number(windowSeconds);
      const windowSamples = validSamples.filter(sample => sample.t_s >= cutoffS);

      if (windowSamples.length < 2) {
        return null;
      }

      const firstTimeS = windowSamples[0].t_s;
      const lastTimeS = windowSamples[windowSamples.length - 1].t_s;
      const durationS = lastTimeS - firstTimeS;

      if (!Number.isFinite(durationS) || durationS <= 0) {
        return null;
      }

      return (windowSamples.length - 1) / durationS;
    }

    async function renderLowFpsModelSkip({
      effectiveFps,
      minimumFpsHz,
      expectedRevision = null,
    } = {}) {
      if (expectedRevision !== null && expectedRevision !== measurementRevision) {
        return null;
      }

      const statusEl = document.getElementById("camera-status");
      const fpsText = effectiveFps === null ? "unknown" : `${effectiveFps.toFixed(2)} Hz`;

      const preservedAnalysis =
        latestRoiAnalysisDisplayState !== null &&
        latestRoiAnalysisDisplayState.revision === measurementRevision &&
        latestRoiAnalysisDisplayState.sampleCount === roiSamples.length
          ? latestRoiAnalysisDisplayState
          : null;

      const primarySpectralHrText =
        preservedAnalysis !== null ? preservedAnalysis.spectralHr : "Not available";

      const primarySpectralDetail =
        preservedAnalysis !== null
          ? preservedAnalysis.spectralDetail
          : "Full-buffer spectral analysis should run before model prediction.";

      const primaryQuality =
        preservedAnalysis !== null ? preservedAnalysis.quality : "Not available";

      const primaryQualityDetail =
        preservedAnalysis !== null
          ? preservedAnalysis.qualityDetail
          : "Signal quality was not available before model skip.";

      const skipDetail =
        `Experimental model skipped because live ROI sampling was ${fpsText}. ` +
        `Training-style 0.7-3.5 Hz bandpass preprocessing needs at least ` +
        `${minimumFpsHz.toFixed(1)} Hz source sampling.`;

      await renderMeasurementResultCardsFromServer({
        spectralHr: primarySpectralHrText,
        spectralDetail: primarySpectralDetail,
        modelHr: "Skipped",
        modelDetail: skipDetail,
        modelDifference: "Not available",
        modelDifferenceDetail: "Agreement diagnostic unavailable because model prediction was skipped.",
        quality: primaryQuality,
        qualityDetail: primaryQualityDetail,
      });

      await renderModelPredictionSummaryFromServer({
        status: "Skipped",
        modelHr: "Skipped",
        spectralConsensus: primarySpectralHrText,
        modelDifference: "Not available",
        greenSummary: "Model-side spectral summary not computed",
        posSummary: "Model-side spectral summary not computed",
        chromSummary: "Model-side spectral summary not computed",
        originalDurationS: "none",
        usedDurationS: "none",
        usedSamples: String(roiSamples.length),
        sourceFps: fpsText,
        rawResponse: JSON.stringify(
          {
            status: "skipped",
            reason: "live_roi_sampling_fps_too_low",
            effective_fps_hz: effectiveFps,
            minimum_required_fps_hz: minimumFpsHz,
            bandpass_high_hz: 3.5,
            message: skipDetail,
          },
          null,
          2
        ),
      });

      if (statusEl) {
        statusEl.innerText =
          preservedAnalysis !== null
            ? `Model skipped. Kept full-buffer spectral estimate (${preservedAnalysis.spectralHr}). ${skipDetail}`
            : `Model skipped. ${skipDetail}`;
      }

      return {
        summary: "Model skipped.",
        detail:
          preservedAnalysis !== null
            ? `Kept full-buffer spectral estimate (${preservedAnalysis.spectralHr}). ${skipDetail}`
            : skipDetail,
      };
    }

    
    async function renderRejectedSignalModelSkip({ expectedRevision = null } = {}) {
      if (expectedRevision !== null && expectedRevision !== measurementRevision) {
        return null;
      }

      const statusEl = document.getElementById("camera-status");

      const preservedAnalysis =
        latestRoiAnalysisDisplayState !== null &&
        latestRoiAnalysisDisplayState.revision === measurementRevision &&
        latestRoiAnalysisDisplayState.sampleCount === roiSamples.length
          ? latestRoiAnalysisDisplayState
          : null;

      const skipDetail =
        "Experimental model not run because the full-buffer spectral quality gate rejected this measurement.";

      await renderMeasurementResultCardsFromServer({
        spectralHr: preservedAnalysis?.spectralHr ?? "Rejected",
        spectralDetail:
          preservedAnalysis?.spectralDetail ??
          "Full-buffer spectral quality gate rejected this measurement.",
        modelHr: "Not run",
        modelDetail: skipDetail,
        modelDifference: "Not available",
        modelDifferenceDetail: "Agreement diagnostic unavailable because model prediction was not run.",
        quality: preservedAnalysis?.quality ?? "Rejected",
        qualityDetail:
          preservedAnalysis?.qualityDetail ??
          "Full-buffer spectral quality gate rejected this measurement.",
      });

      await renderModelPredictionSummaryFromServer({
        status: "Not run",
        modelHr: "Not run",
        spectralConsensus: preservedAnalysis?.diagnosticSpectralHr ?? "Rejected",
        modelDifference: "Not available",
        greenSummary: "Model-side spectral summary not computed",
        posSummary: "Model-side spectral summary not computed",
        chromSummary: "Model-side spectral summary not computed",
        originalDurationS: "none",
        usedDurationS: "none",
        usedSamples: String(roiSamples.length),
        sourceFps: "none",
        rawResponse: JSON.stringify(
          {
            status: "not_run",
            reason: "full_buffer_spectral_quality_rejected",
            message: skipDetail,
          },
          null,
          2
        ),
      });

      if (statusEl) {
        statusEl.innerText =
          preservedAnalysis !== null
            ? `Model not run. Kept rejected full-buffer spectral state. ${skipDetail}`
            : `Model not run. ${skipDetail}`;
      }

      return {
        summary: "Model not run.",
        detail: skipDetail,
      };
    }


    async function runLiveModelPredictionInBackend(expectedRevision = null) {
      if (typeof expectedRevision !== "number") {
        expectedRevision = null;
      }

      const statusEl = document.getElementById("camera-status");
      const runButton = document.getElementById("run-live-model-button");

      if (!window.livePredictionRuns) {
        window.livePredictionRuns = [];
      }

      if (roiSamples.length < 20) {
        const detail = "Collect at least 20 ROI samples before live model prediction.";
        statusEl.innerText = detail;
        return {
          summary: "Prediction unavailable.",
          detail: detail,
        };
      }

      const minimumModelSourceFpsHz = 8.0;
      const effectiveModelFps = estimateRoiSamplesEffectiveFpsForModel(roiSamples, 12.0);

      if (effectiveModelFps === null || effectiveModelFps < minimumModelSourceFpsHz) {
        return await renderLowFpsModelSkip({
          effectiveFps: effectiveModelFps,
          minimumFpsHz: minimumModelSourceFpsHz,
          expectedRevision: expectedRevision,
        });
      }

      const shouldRespectFullBufferRejection = expectedRevision !== null;
      const preservedAnalysisForModelGate =
        latestRoiAnalysisDisplayState !== null &&
        latestRoiAnalysisDisplayState.revision === measurementRevision &&
        latestRoiAnalysisDisplayState.sampleCount === roiSamples.length
          ? latestRoiAnalysisDisplayState
          : null;

      if (
        shouldRespectFullBufferRejection &&
        preservedAnalysisForModelGate?.quality === "Rejected"
      ) {
        return await renderRejectedSignalModelSkip({
          expectedRevision: expectedRevision,
        });
      }

      if (runButton) {
        runButton.disabled = true;
        runButton.innerText = "Predicting...";
      }

      try {
        const response = await fetch("/api/predict-live-roi-series", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            samples: roiSamples
          })
        });

        const data = await parseJsonResponseEvenOnError(response);

        if (expectedRevision !== null && expectedRevision !== measurementRevision) {
          return null;
        }

        const summary = summarizeModelPrediction(data);
        const modelUnavailable = isModelUnavailableResponse(data);
        const modelRejected = !modelUnavailable && summary.qualityStatus === "rejected";
        const quality = updateMeasurementQualityFromModelPrediction(data);

        const preservedAnalysis =
          latestRoiAnalysisDisplayState !== null &&
          latestRoiAnalysisDisplayState.revision === measurementRevision &&
          latestRoiAnalysisDisplayState.sampleCount === roiSamples.length
            ? latestRoiAnalysisDisplayState
            : null;

        let modelHrText = modelUnavailable
          ? "Unavailable"
          : modelRejected
            ? "Rejected"
            : formatBpm(summary.modelHr);

        let modelDetail = modelUnavailable
          ? modelUnavailableDetail(data)
          : modelRejected
            ? "Experimental model did not return HR because the model-window quality gate rejected this input."
            : "Experimental CRVSE PhysFormer output";

        let modelSummaryHrText = modelHrText;

        const modelWindowSpectralText = modelRejected
          ? "Unavailable"
          : formatBpm(summary.consensus);

        const modelWindowSpectralDetail = modelRejected
          ? "Model-window spectral gate rejected this window; no primary HR from this model run."
          : "Model-window spectral consensus from latest model input window.";

        const primarySpectralHrText =
          preservedAnalysis !== null
            ? preservedAnalysis.spectralHr
            : modelWindowSpectralText;

        const primarySpectralDetail =
          preservedAnalysis !== null
            ? preservedAnalysis.spectralDetail
            : modelWindowSpectralDetail;

        const primaryQuality =
          preservedAnalysis !== null
            ? preservedAnalysis.quality
            : quality.summary;

        const primaryQualityDetail =
          preservedAnalysis !== null
            ? preservedAnalysis.qualityDetail
            : quality.detail;

        const modelVsWindowDifferenceForDisplay =
          modelUnavailable || modelRejected
            ? null
            : getValidNumber(summary.difference);

        let modelDifferenceText = modelUnavailable || modelRejected
          ? "Not available"
          : formatSignedBpm(summary.difference);

        let modelDifferenceDetail = modelUnavailable
          ? "Agreement diagnostic unavailable because model prediction is unavailable."
          : modelRejected
            ? "Agreement diagnostic unavailable because the model-window spectral gate rejected the input."
            : modelSpectralAgreementDetail(summary.difference);

        if (
          modelVsWindowDifferenceForDisplay !== null &&
          Math.abs(modelVsWindowDifferenceForDisplay) >= 10.0
        ) {
          const rawModelHrText = formatBpm(summary.modelHr);
          const modelWindowText = formatBpm(summary.consensus);

          modelHrText = "Disagrees";
          modelSummaryHrText = rawModelHrText;
          modelDetail =
            `Raw model HR ${rawModelHrText}; model-window spectral ${modelWindowText}; ` +
            `difference ${formatSignedBpm(modelVsWindowDifferenceForDisplay)}. Treat model HR as experimental.`;
          modelDifferenceDetail =
            "Large model-vs-spectral disagreement. Spectral HR remains the primary estimate.";
        }
        await renderMeasurementResultCardsFromServer({
          spectralHr: primarySpectralHrText,
          spectralDetail: primarySpectralDetail,
          modelHr: modelHrText,
          modelDetail: modelDetail,
          modelDifference: modelDifferenceText,
          modelDifferenceDetail: modelDifferenceDetail,
          quality: primaryQuality,
          qualityDetail: primaryQualityDetail
        });

        await renderModelPredictionSummaryFromServer({
          status: modelUnavailable
            ? "Model unavailable"
            : modelRejected
              ? "Rejected"
              : data.status ?? "unknown",
          modelHr: modelSummaryHrText,
          spectralConsensus: modelWindowSpectralText,
          modelDifference: modelDifferenceText,
          greenSummary: `${formatBpm(summary.greenBpm)} / SQI ${formatNumber(summary.greenSqi, 3)} / ${summary.greenStatus}`,
          posSummary: `${formatBpm(summary.posBpm)} / SQI ${formatNumber(summary.posSqi, 3)} / ${summary.posStatus}`,
          chromSummary: `${formatBpm(summary.chromBpm)} / SQI ${formatNumber(summary.chromSqi, 3)} / ${summary.chromStatus}`,
          originalDurationS: formatSecondsOrNone(summary.originalDurationS),
          usedDurationS: formatSecondsOrNone(summary.usedDurationS),
          usedSamples: String(summary.usedSamples ?? "none"),
          sourceFps: formatHzOrNone(summary.sourceFps),
          rawResponse: JSON.stringify(buildCompactModelPredictionDebugResponse(data), null, 2),
        });

        if (expectedRevision !== null && expectedRevision !== measurementRevision) {
          return null;
        }

        if (modelUnavailable) {
          const detail =
            preservedAnalysis !== null
              ? `Kept primary full-buffer spectral estimate (${preservedAnalysis.spectralHr}). Experimental model unavailable.`
              : `Experimental model unavailable. Model-window spectral consensus=${formatBpm(summary.consensus)}.`;

          statusEl.innerText = detail;

          return {
            summary: "Model unavailable.",
            detail: detail,
          };
        }

        await addPredictionRun(summary);

        if (modelRejected) {
          const detail =
            preservedAnalysis !== null
              ? `Kept primary full-buffer spectral estimate (${preservedAnalysis.spectralHr}). Model-window spectral gate rejected this window.`
              : "Model-window spectral gate rejected this window. Review channel SQI and spread diagnostics.";

          statusEl.innerText =
            "Live model prediction rejected by model-window spectral gate. Review channel SQI and spread diagnostics.";

          return {
            summary: "Prediction rejected.",
            detail: detail,
          };
        }

        const modelWindowSpectralBpm = getValidNumber(summary.consensus);
        const fullBufferSpectralBpm =
          preservedAnalysis !== null
            ? getValidNumber(preservedAnalysis.spectralBpm)
            : null;

        const modelWindowVsFullBufferDifference =
          modelWindowSpectralBpm !== null && fullBufferSpectralBpm !== null
            ? modelWindowSpectralBpm - fullBufferSpectralBpm
            : null;

        const modelVsWindowDifference = getValidNumber(summary.difference);
        const absoluteModelVsWindowDifference = Math.abs(modelVsWindowDifference ?? 0.0);
        const absoluteWindowVsFullBufferDifference = Math.abs(modelWindowVsFullBufferDifference ?? 0.0);

        const modelText = formatBpm(summary.modelHr);
        const modelWindowSpectralStatusText = formatBpm(summary.consensus);
        const fullBufferSpectralText =
          preservedAnalysis !== null
            ? preservedAnalysis.spectralHr
            : "none";

        statusEl.innerText =
          `Live model prediction completed. Model=${modelText}, ` +
          `model-window spectral consensus=${modelWindowSpectralStatusText}.`;

        if (modelWindowVsFullBufferDifference !== null && absoluteWindowVsFullBufferDifference >= 15.0) {
          return {
            summary: "Prediction complete with window disagreement.",
            detail:
              `Full-buffer spectral=${fullBufferSpectralText}, ` +
              `model-window spectral=${modelWindowSpectralStatusText}, ` +
              `difference=${formatSignedBpm(modelWindowVsFullBufferDifference)}. ` +
              "Keep the full-buffer spectral estimate as primary and treat model-window/model output as experimental.",
          };
        }

        if (absoluteModelVsWindowDifference >= 10.0) {
          return {
            summary: "Prediction complete with model disagreement.",
            detail:
              `Model=${modelText}, model-window spectral=${modelWindowSpectralStatusText}, ` +
              `difference=${formatSignedBpm(modelVsWindowDifference)}. Treat model HR as experimental.`,
          };
        }

        return {
          summary: "Prediction complete.",
          detail:
            `Primary full-buffer spectral=${primarySpectralHrText}. ` +
            `Model=${modelText}, model-window spectral=${modelWindowSpectralStatusText}. ` +
            "Review the HR cards and waveform. Use Clear before a new measurement if needed.",
        };
      } catch (error) {
        const modelError = `${error}`;
        const preservedAnalysis =
          latestRoiAnalysisDisplayState !== null &&
          latestRoiAnalysisDisplayState.revision === measurementRevision &&
          latestRoiAnalysisDisplayState.sampleCount === roiSamples.length
            ? latestRoiAnalysisDisplayState
            : null;

        statusEl.innerText = `Live model prediction failed: ${modelError}`;

        if (preservedAnalysis !== null) {
          await renderMeasurementResultCardsFromServer({
            spectralHr: preservedAnalysis.spectralHr,
            spectralDetail: preservedAnalysis.spectralDetail,
            modelHr: "Unavailable",
            modelDetail: "Experimental model prediction unavailable for this run.",
            modelDifference: "Not available",
            modelDifferenceDetail: "Agreement diagnostic unavailable because model prediction failed.",
            quality: preservedAnalysis.quality,
            qualityDetail: preservedAnalysis.qualityDetail,
          });

          await renderModelPredictionSummaryFromServer({
            status: "Prediction unavailable",
            modelHr: "none",
            spectralConsensus: "none",
            modelDifference: "none",
            greenSummary: "GREEN model-side spectral summary unavailable",
            posSummary: "POS model-side spectral summary unavailable",
            chromSummary: "CHROM model-side spectral summary unavailable",
            originalDurationS: "none",
            usedDurationS: "none",
            usedSamples: "none",
            sourceFps: "none",
            rawResponse: modelError,
          });

          return {
            summary: "Model prediction unavailable.",
            detail:
              `Kept full-buffer spectral estimate (${preservedAnalysis.spectralHr}). ` +
              `Model error: ${modelError}`,
          };
        }

        await renderMeasurementResultCardsFromServer({
          spectralHr: "Prediction failed",
          spectralDetail: "Model-window spectral consensus unavailable after prediction failure",
          modelHr: "Prediction failed",
          modelDetail: "Experimental model prediction failed",
          modelDifference: "Prediction failed",
          modelDifferenceDetail: "Agreement diagnostic unavailable",
          quality: "Prediction failed",
          qualityDetail: "Model prediction failed before quality summary update.",
        });

        await renderModelPredictionSummaryFromServer({
          status: "Prediction failed",
          modelHr: "none",
          spectralConsensus: "none",
          modelDifference: "none",
          greenSummary: "GREEN model-side spectral summary unavailable",
          posSummary: "POS model-side spectral summary unavailable",
          chromSummary: "CHROM model-side spectral summary unavailable",
          originalDurationS: "none",
          usedDurationS: "none",
          usedSamples: "none",
          sourceFps: "none",
          rawResponse: modelError,
        });

        return {
          summary: "Prediction failed.",
          detail: modelError,
        };
      } finally {
        if (runButton) {
          runButton.disabled = false;
          runButton.innerText = "Run live model prediction";
        }
      }
    }

    document.addEventListener("DOMContentLoaded", () => {
      const startCameraButton = document.getElementById("start-camera-button");
      const captureFrameButton = document.getElementById("capture-frame-button");
      const sendFrameButton = document.getElementById("send-frame-button");
      const detectFaceButton = document.getElementById("detect-face-button");
      const stopCameraButton = document.getElementById("stop-camera-button");
      const startMeasurementButton = document.getElementById("start-measurement-button");
      const stopMeasurementButton = document.getElementById("stop-measurement-button");
      const startRoiSamplingButton = document.getElementById("start-roi-sampling-button");
      const stopRoiSamplingButton = document.getElementById("stop-roi-sampling-button");
      const clearRoiSamplesButton = document.getElementById("clear-roi-samples-button");
      const analyzeRoiSeriesButton = document.getElementById("analyze-roi-series-button");
      const runLiveModelButton = document.getElementById("run-live-model-button");

      drawAllRoiPlots();
      drawMainPulseWaveformPlaceholder();
      resetMeasurementProgress();
      resetFinalInterpretationPanel();

      if (startCameraButton) {
        startCameraButton.addEventListener("click", startCameraPreview);
      }

      if (captureFrameButton) {
        captureFrameButton.addEventListener("click", captureOneFrame);
      }

      if (sendFrameButton) {
        sendFrameButton.addEventListener("click", sendCapturedFrameToBackend);
      }

      if (detectFaceButton) {
        detectFaceButton.addEventListener("click", detectFaceInBackend);
      }

      if (stopCameraButton) {
        stopCameraButton.addEventListener("click", stopCameraPreview);
      }

      if (startMeasurementButton) {
        startMeasurementButton.addEventListener("click", startMainMeasurement);
      }

      if (stopMeasurementButton) {
        stopMeasurementButton.addEventListener("click", stopMainMeasurement);
      }

      if (startRoiSamplingButton) {
        startRoiSamplingButton.addEventListener("click", startRoiSampling);
      }

      if (stopRoiSamplingButton) {
        stopRoiSamplingButton.addEventListener("click", stopRoiSampling);
      }

      if (clearRoiSamplesButton) {
        clearRoiSamplesButton.addEventListener("click", clearRoiSamples);
      }

      if (analyzeRoiSeriesButton) {
        analyzeRoiSeriesButton.addEventListener("click", analyzeRoiSeriesInBackend);
      }

      if (runLiveModelButton) {
        runLiveModelButton.addEventListener("click", runLiveModelPredictionInBackend);
      }

      updatePrimaryControlButtons();
    });
