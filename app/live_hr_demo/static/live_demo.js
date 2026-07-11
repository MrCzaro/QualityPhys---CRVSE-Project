  let cameraStream = null;
  let hasCapturedFrame = false;
  let roiSamplingTimer = null;
  let roiSamples = [];
  let roiSamplingStartMs = null;
  let roiSamplingInFlight = false;
  let roiSamplingRunId = 0;
  let mainMeasurementTimer = null;
  let mainMeasurementProgressTimer = null;
  let mainMeasurementInProgress = false;
  let measurementRevision = 0;

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

      const hasRevisionGuard = typeof expectedRevision === "number";

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

      The full backend response can contain model input arrays or verbose metadata.
      The server-rendered panel receives the clinically useful summary plus a compact
      debug payload, keeping URL size reasonable.
      */

      return {
        status: data.status,
        message: data.message ?? null,
        model_prediction: data.model_prediction ?? null,
        classical_spectral_summary: data.classical_spectral_summary ?? null,
        model_input: {
          window_metadata: data.model_input?.window_metadata ?? null,
          source_estimated_fps: data.model_input?.source_estimated_fps ?? null,
          channel_names: data.model_input?.channel_names ?? null,
          target_frames: data.model_input?.target_frames ?? null,
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

    function removeRepeatabilityTable() {
      const repeatabilityContainer = document.getElementById("live-model-repeatability-container");

      if (repeatabilityContainer) {
        repeatabilityContainer.remove();
      }
    }

    function resetMeasurementOutputs(expectedRevision = null) {
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
      removeRepeatabilityTable();
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
        const elapsedSeconds = Math.min(
          MAIN_MEASUREMENT_DURATION_MS / 1000.0,
          Math.max(0.0, elapsedMs / 1000.0)
        );
        const totalSeconds = MAIN_MEASUREMENT_DURATION_MS / 1000.0;

        const phaseText = elapsedSeconds < 3.0
          ? "Stabilizing signal. Hold still."
          : "Collecting rPPG signal. Keep your face steady.";

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


    function setMainMeasurementButtonsState(isRunning) {
      const startMeasurementButton = document.getElementById("start-measurement-button");
      const stopMeasurementButton = document.getElementById("stop-measurement-button");

      if (startMeasurementButton) {
        startMeasurementButton.disabled = isRunning;
        startMeasurementButton.innerText = isRunning ? "Measuring..." : "Start measurement";
      }

      if (stopMeasurementButton) {
        stopMeasurementButton.disabled = !isRunning;
      }
    }

    function mean(values) {
      if (values.length === 0) {
        return null;
      }

      return values.reduce((sum, value) => sum + value, 0) / values.length;
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

      startButton.disabled = true;
      startButton.innerText = "Starting...";

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

        statusEl.innerText =
          "Camera started. Preview is local in the browser. Frames are not sent to the backend automatically.";
      } catch (error) {
        statusEl.innerText = `Camera start failed: ${error}`;
      } finally {
        startButton.disabled = false;
        startButton.innerText = "Start camera";
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

      mainMeasurementInProgress = false;
      measurementRevision += 1;
      stopMeasurementProgressTimer();
      setMainMeasurementButtonsState(false);
      setMeasurementStatus("Camera stopped.", "Start the camera again before running a new measurement.");

      if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
      }

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

      statusEl.innerText = "Camera stopped. Captured frame remains local in the canvas.";
    }

    function captureOneFrame() {
      const videoEl = document.getElementById("camera-video");
      const canvasEl = document.getElementById("snapshot-canvas");
      const statusEl = document.getElementById("camera-status");

      if (!cameraStream) {
        statusEl.innerText = "Cannot capture frame: camera is not started.";
        return false;
      }

      const width = videoEl.videoWidth;
      const height = videoEl.videoHeight;

      if (width === 0 || height === 0) {
        statusEl.innerText = "Cannot capture frame yet: video dimensions are not ready.";
        return false;
      }

      canvasEl.width = width;
      canvasEl.height = height;

      const context = canvasEl.getContext("2d");
      context.drawImage(videoEl, 0, 0, width, height);

      hasCapturedFrame = true;

      statusEl.innerText =
        `Captured one local frame: ${width} x ${height}. Frame was not sent to backend.`;

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

      const width = videoEl.videoWidth;
      const height = videoEl.videoHeight;

      if (width === 0 || height === 0) {
        statusEl.innerText = "Cannot capture sampling frame yet: video dimensions are not ready.";
        return null;
      }

      canvasEl.width = width;
      canvasEl.height = height;

      const context = canvasEl.getContext("2d");
      context.drawImage(videoEl, 0, 0, width, height);

      return canvasEl.toDataURL("image/jpeg", 0.85);
    }

    function extractRoiSampleFromBackendResponse(data) {
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
        return;
      }

      const firstT = roiSamples[0].t_s;
      const lastT = roiSamples[roiSamples.length - 1].t_s;
      const durationS = Math.max(0, lastT - firstT);

      const lines = [];

      lines.push(`samples: ${roiSamples.length}`);
      lines.push(`duration_s: ${durationS.toFixed(2)}`);
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
    }

    async function collectOneRoiSample(
      expectedSamplingRunId = roiSamplingRunId,
      expectedRevision = measurementRevision
    ) {
      const statusEl = document.getElementById("camera-status");

      if (roiSamplingInFlight) {
        return;
      }

      roiSamplingInFlight = true;

      try {
        const imageDataUrl = captureFrameForSamplingDataUrl();

        if (!imageDataUrl) {
          return;
        }

        if (
          expectedSamplingRunId !== roiSamplingRunId ||
          expectedRevision !== measurementRevision
        ) {
          return;
        }

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

        if (
          expectedSamplingRunId !== roiSamplingRunId ||
          expectedRevision !== measurementRevision ||
          roiSamplingTimer === null
        ) {
          return;
        }

        const sample = extractRoiSampleFromBackendResponse(data);

        roiSamples.push(sample);

        summarizeCollectedRoiSamples();

        statusEl.innerText =
          `ROI sampling active. Collected ${roiSamples.length} sample(s). ` +
          "Frames are processed in memory and not stored.";
      } catch (error) {
        if (
          expectedSamplingRunId === roiSamplingRunId &&
          expectedRevision === measurementRevision &&
          roiSamplingTimer !== null
        ) {
          statusEl.innerText = `ROI sampling error: ${error}`;
        }
      } finally {
        if (expectedSamplingRunId === roiSamplingRunId) {
          roiSamplingInFlight = false;
        }
      }
    }

    function startRoiSampling({ resetOutputs = true } = {}) {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      const samplingIntervalMs = 100;

      if (!cameraStream) {
        statusEl.innerText = "Cannot start ROI sampling: camera is not started.";
        return false;
      }

      if (roiSamplingTimer !== null) {
        statusEl.innerText = "ROI sampling is already running.";
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
        `ROI sampling started at ${samplingIntervalMs} ms interval. Hold still for about 8-10 seconds.`;

      summarizeCollectedRoiSamples();

      roiSamplingTimer = setInterval(() => {
        collectOneRoiSample(activeSamplingRunId, activeRevision);
      }, samplingIntervalMs);

      collectOneRoiSample(activeSamplingRunId, activeRevision);

      return true;
    }


    function stopRoiSampling() {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      roiSamplingRunId += 1;
      roiSamplingInFlight = false;

      if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
      }

      if (startButton) {
        startButton.disabled = false;
        startButton.innerText = "Start ROI sampling";
      }

      summarizeCollectedRoiSamples();

      statusEl.innerText =
        `ROI sampling stopped. Collected ${roiSamples.length} sample(s).`;
    }

    function clearRoiSamples() {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      if (mainMeasurementTimer !== null) {
        clearTimeout(mainMeasurementTimer);
        mainMeasurementTimer = null;
      }

      mainMeasurementInProgress = false;
      measurementRevision += 1;
      const activeRevision = measurementRevision;
      roiSamplingRunId += 1;

      stopMeasurementProgressTimer();
      setMainMeasurementButtonsState(false);
      setMeasurementStatus("Camera stopped.", "Start the camera again before running a new measurement.");

      if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
      }

      roiSamples = [];
      roiSamplingStartMs = null;
      roiSamplingInFlight = false;

      window.livePredictionRuns = [];

      if (startButton) {
        startButton.disabled = false;
        startButton.innerText = "Start ROI sampling";
      }

      resetMeasurementOutputs(activeRevision);
      drawMainPulseWaveformPlaceholder();

      if (statusEl) {
        statusEl.innerText = "ROI samples and measurement results cleared.";
      }
    }

    async function startMainMeasurement() {
      const statusEl = document.getElementById("camera-status");

      if (!cameraStream) {
        statusEl.innerText = "Cannot start measurement: camera is not started.";
        return;
      }

      if (mainMeasurementInProgress) {
        statusEl.innerText = "Measurement is already running.";
        return;
      }

      clearRoiSamples();

      measurementRevision += 1;
      const activeRevision = measurementRevision;

      mainMeasurementInProgress = true;
      setMainMeasurementButtonsState(true);

      const measurementStartedAtMs = performance.now();
      setMeasurementStatus(
        "Measuring...",
        `Stabilizing signal. 0.0 / ${(MAIN_MEASUREMENT_DURATION_MS / 1000).toFixed(1)} s`
      );
      setMeasurementProgress(0.0, `0.0 / ${(MAIN_MEASUREMENT_DURATION_MS / 1000).toFixed(1)} s`);
      startMeasurementProgressTimer(activeRevision, measurementStartedAtMs);

      const samplingStarted = startRoiSampling({
        resetOutputs: false,
      });

      if (!samplingStarted || roiSamplingTimer === null) {
        mainMeasurementInProgress = false;
        stopMeasurementProgressTimer();
        setMainMeasurementButtonsState(false);
        setMeasurementStatus("Measurement could not start.", "Camera or video frame was not ready.");
        setMeasurementProgress(0.0, "0%");
        return;
      }

      statusEl.innerText =
        `Measurement started. Hold still for ${(MAIN_MEASUREMENT_DURATION_MS / 1000).toFixed(0)} seconds.`;

      mainMeasurementTimer = setTimeout(async () => {
        mainMeasurementTimer = null;
        stopMeasurementProgressTimer();

        try {
          if (activeRevision !== measurementRevision) {
            return;
          }

          setMeasurementProgress(1.0, `${(MAIN_MEASUREMENT_DURATION_MS / 1000).toFixed(1)} / ${(MAIN_MEASUREMENT_DURATION_MS / 1000).toFixed(1)} s`);
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

          await runLiveModelPredictionInBackend(activeRevision);

          if (activeRevision === measurementRevision) {
            setMeasurementStatus(
              "Prediction complete.",
              "Review the HR cards and waveform. Use Clear before a new measurement if needed."
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
          }
        }
      }, MAIN_MEASUREMENT_DURATION_MS);
    }

    function stopMainMeasurement() {
      const statusEl = document.getElementById("camera-status");

      if (mainMeasurementTimer !== null) {
        clearTimeout(mainMeasurementTimer);
        mainMeasurementTimer = null;
      }

      measurementRevision += 1;
      stopMeasurementProgressTimer();

      if (roiSamplingTimer !== null) {
        stopRoiSampling();
      }

      mainMeasurementInProgress = false;
      setMainMeasurementButtonsState(false);

      statusEl.innerText =
        `Measurement stopped. Collected ${roiSamples.length} sample(s). Camera is still active.`;
      setMeasurementStatus(
        "Measurement stopped.",
        `Collected ${roiSamples.length} sample(s). You can analyze manually from diagnostics or clear and retry.`
      );
      setMeasurementProgress(0.0, "Stopped");
    }

    function summarizeSpectralQualityFromEntries(entries) {
      const validEntries = entries.filter(entry => entry !== null);

      if (validEntries.length === 0) {
        return {
          summary: "Not available",
          detail: "No spectral channel results returned."
        };
      }

      const bpmValues = validEntries
        .map(entry => getValidNumber(entry.bpm))
        .filter(value => value !== null);

      const goodEntries = validEntries.filter(entry => entry.status === "good");
      const moderateEntries = validEntries.filter(entry => entry.status === "moderate");
      const usableEntries = goodEntries.concat(moderateEntries);

      if (bpmValues.length === 0) {
        return {
          summary: "Rejected",
          detail: "No dominant HR peak detected in the cardiac band."
        };
      }

      const bpmMin = Math.min(...bpmValues);
      const bpmMax = Math.max(...bpmValues);
      const bpmSpread = bpmMax - bpmMin;
      const maxAllowedSpread = 20.0;

      if (bpmSpread > maxAllowedSpread) {
        return {
          summary: "Rejected",
          detail: `Channel HR peaks disagree: spread ${bpmSpread.toFixed(1)} bpm.`
        };
      }

      if (goodEntries.length > 0) {
        return {
          summary: "Accepted / good",
          detail: `${goodEntries.length} good channel(s), spread ${bpmSpread.toFixed(1)} bpm.`
        };
      }

      if (usableEntries.length > 0) {
        return {
          summary: "Accepted / moderate",
          detail: `${usableEntries.length} moderate channel(s), spread ${bpmSpread.toFixed(1)} bpm.`
        };
      }

      return {
        summary: "Rejected",
        detail: "No channel reached moderate or good spectral quality."
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

      return summarizeSpectralQualityFromEntries(entries);
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

      return summarizeSpectralQualityFromEntries(entries);
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
        return;
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
          return;
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

        await renderSignalSummaryCardsFromServer({
          green: formatSignalSummary("GREEN", data.signals?.green),
          greenDetail: formatSignalDetail("GREEN", data.signals?.green),
          pos: formatSignalSummary("POS", data.signals?.pos),
          posDetail: formatSignalDetail("POS", data.signals?.pos),
          chrom: formatSignalSummary("CHROM", data.signals?.chrom),
          chromDetail: formatSignalDetail("CHROM", data.signals?.chrom),
        });

        await renderMeasurementResultCardsFromServer({
          spectralHr: formatBpm(spectralConsensus),
          spectralDetail: "Primary estimate: spectral consensus from GREEN / POS / CHROM",
          modelHr: "Not predicted yet",
          modelDetail: "Run live model prediction to compare with spectral HR",
          modelDifference: "Not predicted yet",
          modelDifferenceDetail: "Agreement diagnostic",
          quality: quality.summary,
          qualityDetail: quality.detail,
        });

        await renderRoiAnalysisSummaryFromServer({
          status: data.status ?? "unknown",
          sampleCount: String(data.sample_count ?? "none"),
          durationS: `${formatNumber(data.duration_s, 2)} s`,
          estimatedFps: `${formatNumber(data.estimated_fps, 2)} Hz`,
          spectralConsensus: formatBpm(spectralConsensus),
          greenSummary: formatSignalSummary("GREEN", data.signals?.green),
          posSummary: formatSignalSummary("POS", data.signals?.pos),
          chromSummary: formatSignalSummary("CHROM", data.signals?.chrom),
          rawResponse: JSON.stringify(buildCompactRoiAnalysisDebugResponse(data), null, 2),
        });

        statusEl.innerText =
          `ROI series analyzed. Spectral consensus=${formatBpm(spectralConsensus)}. ` +
          `GREEN=${greenBpm?.toFixed(1) ?? "none"} BPM, ` +
          `POS=${posBpm?.toFixed(1) ?? "none"} BPM, ` +
          `CHROM=${chromBpm?.toFixed(1) ?? "none"} BPM.`;
      } catch (error) {
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
          spectralDetail: "Spectral HR could not be computed",
          modelHr: "Not predicted yet",
          modelDetail: "Model prediction was not run",
          modelDifference: "Not predicted yet",
          modelDifferenceDetail: "Agreement diagnostic unavailable",
          quality: "Analysis failed",
          qualityDetail: "Signal quality could not be computed.",
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

    function summarizeModelPrediction(data) {
      const modelHr = getValidNumber(data.model_prediction?.value);
      const consensus = mean(spectralBpmValues(data));
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
        statusEl.innerText = "Collect at least 20 ROI samples before live model prediction.";
        return;
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
          return;
        }

        const summary = summarizeModelPrediction(data);

        const quality = updateMeasurementQualityFromModelPrediction(data);

        await renderMeasurementResultCardsFromServer({
          spectralHr: formatBpm(summary.consensus),
          spectralDetail: "Primary estimate: spectral consensus",
          modelHr: formatBpm(summary.modelHr),
          modelDetail: "Experimental CRVSE PhysFormer output",
          modelDifference: formatSignedBpm(summary.difference),
          modelDifferenceDetail: "Agreement diagnostic",
          quality: quality.summary,
          qualityDetail: quality.detail
        });

        await renderModelPredictionSummaryFromServer({
          status: data.status ?? "unknown",
          modelHr: formatBpm(summary.modelHr),
          spectralConsensus: formatBpm(summary.consensus),
          modelDifference: formatSignedBpm(summary.difference),
          greenSummary: `${formatBpm(summary.greenBpm)} / SQI ${formatNumber(summary.greenSqi, 3)} / ${summary.greenStatus}`,
          posSummary: `${formatBpm(summary.posBpm)} / SQI ${formatNumber(summary.posSqi, 3)} / ${summary.posStatus}`,
          chromSummary: `${formatBpm(summary.chromBpm)} / SQI ${formatNumber(summary.chromSqi, 3)} / ${summary.chromStatus}`,
          originalDurationS: `${formatNumber(summary.originalDurationS, 2)} s`,
          usedDurationS: `${formatNumber(summary.usedDurationS, 2)} s`,
          usedSamples: String(summary.usedSamples ?? "none"),
          sourceFps: `${formatNumber(summary.sourceFps, 2)} Hz`,
          rawResponse: JSON.stringify(buildCompactModelPredictionDebugResponse(data), null, 2),
        });

        if (expectedRevision !== null && expectedRevision !== measurementRevision) {
          return;
        }

        await addPredictionRun(summary);

        statusEl.innerText =
          `Live model prediction completed. Model=${formatBpm(summary.modelHr)}, ` +
          `spectral consensus=${formatBpm(summary.consensus)}.`;
      } catch (error) {
        statusEl.innerText = `Live model prediction failed: ${error}`;

        await renderMeasurementResultCardsFromServer({
          spectralHr: "Prediction failed",
          spectralDetail: "Spectral consensus unavailable after prediction failure",
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
          rawResponse: `${error}`,
        });
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
        stopMeasurementButton.disabled = true;
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
    });
