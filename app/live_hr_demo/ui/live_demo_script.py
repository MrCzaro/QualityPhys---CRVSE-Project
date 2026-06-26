from fasthtml.common import FT, NotStr, Script


def live_demo_script() -> FT:
    """
    Return browser-side JavaScript for the live HR demo.

    The script manages camera preview, debug frame capture, ROI sampling,
    live waveform drawing, backend rPPG analysis, and experimental model
    prediction updates.

    Returns
    -------
    FT
        FastHTML script element containing the live demo JavaScript.

    Notes
    -----
    ROI sampling uses a hidden canvas to avoid repainting the visible debug
    canvas during acquisition. Image frames are sent to backend debug routes
    for in-memory processing only; the browser stores numeric ROI summaries.
    """

    script = """
    let cameraStream = null;
    let hasCapturedFrame = false;
    let roiSamplingTimer = null;
    let roiSamples = [];
    let roiSamplingStartMs = null;
    let roiSamplingInFlight = false;

    const ROI_NAMES = ["forehead", "image_left_cheek", "image_right_cheek"];

    const ROI_COLORS = {
      forehead: "#7c3aed",
      image_left_cheek: "#16a34a",
      image_right_cheek: "#dc2626"
    };

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

    async function refreshSyntheticResult() {
      const button = document.getElementById("refresh-api-button");
      const modelHrEl = document.getElementById("api-model-hr");
      const spectralHrEl = document.getElementById("api-spectral-hr");
      const qualityEl = document.getElementById("api-quality");
      const reasonEl = document.getElementById("api-reason");

      if (!button || !modelHrEl || !spectralHrEl || !qualityEl || !reasonEl) {
        return;
      }

      button.disabled = true;
      button.innerText = "Refreshing...";

      try {
        const response = await fetch("/api/synthetic-result");

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        const modelHr = data.model_hr_bpm;
        const spectralHr = data.extra?.spectral_hr_bpm;
        const unit = data.unit ?? "bpm";

        modelHrEl.innerText = modelHr === null ? "Unavailable" : `${modelHr.toFixed(1)} ${unit}`;
        spectralHrEl.innerText = spectralHr === null ? "Unavailable" : `${spectralHr.toFixed(1)} ${unit}`;
        qualityEl.innerText = `${data.quality.status} / ${data.quality.confidence}`;

        const reasons = data.quality.reasons ?? [];
        reasonEl.innerText = reasons.length > 0 ? reasons[0] : "No reason returned.";
      } catch (error) {
        modelHrEl.innerText = "Error";
        spectralHrEl.innerText = "Error";
        qualityEl.innerText = "Error";
        reasonEl.innerText = `API call failed: ${error}`;
      } finally {
        button.disabled = false;
        button.innerText = "Refresh synthetic result";
      }
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
        summaryEl.innerText = lines.join("\\n");
      }

      drawAllRoiPlots();
      drawMainPulseWaveformFromLiveSamples();
    }

    async function collectOneRoiSample() {
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
        const sample = extractRoiSampleFromBackendResponse(data);

        roiSamples.push(sample);

        summarizeCollectedRoiSamples();

        statusEl.innerText =
          `ROI sampling active. Collected ${roiSamples.length} sample(s). ` +
          "Frames are processed in memory and not stored.";
      } catch (error) {
        statusEl.innerText = `ROI sampling error: ${error}`;
      } finally {
        roiSamplingInFlight = false;
      }
    }

    function startRoiSampling() {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      const samplingIntervalMs = 100;

      if (!cameraStream) {
        statusEl.innerText = "Cannot start ROI sampling: camera is not started.";
        return;
      }

      if (roiSamplingTimer !== null) {
        statusEl.innerText = "ROI sampling is already running.";
        return;
      }

      roiSamples = [];
      roiSamplingStartMs = performance.now();

      startButton.disabled = true;
      startButton.innerText = "Sampling...";

      statusEl.innerText =
        `ROI sampling started at ${samplingIntervalMs} ms interval. Hold still for about 8-10 seconds.`;

      summarizeCollectedRoiSamples();

      collectOneRoiSample();

      roiSamplingTimer = setInterval(() => {
        collectOneRoiSample();
      }, samplingIntervalMs);
    }

    function stopRoiSampling() {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
      }

      startButton.disabled = false;
      startButton.innerText = "Start ROI sampling";

      summarizeCollectedRoiSamples();

      statusEl.innerText =
        `ROI sampling stopped. Collected ${roiSamples.length} sample(s).`;
    }

    function clearRoiSamples() {
      const statusEl = document.getElementById("camera-status");
      const startButton = document.getElementById("start-roi-sampling-button");

      if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
      }

      roiSamples = [];
      roiSamplingStartMs = null;
      roiSamplingInFlight = false;

      if (startButton) {
        startButton.disabled = false;
        startButton.innerText = "Start ROI sampling";
      }

      summarizeCollectedRoiSamples();

      statusEl.innerText = "ROI samples cleared.";
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

    async function analyzeRoiSeriesInBackend() {
      const statusEl = document.getElementById("camera-status");
      const outputEl = document.getElementById("roi-series-analysis-output");
      const analyzeButton = document.getElementById("analyze-roi-series-button");

      const greenSummaryEl = document.getElementById("green-signal-summary");
      const posSummaryEl = document.getElementById("pos-signal-summary");
      const chromSummaryEl = document.getElementById("chrom-signal-summary");
      const spectralConsensusEl = document.getElementById("spectral-consensus-summary");

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

      function setCompactSignalSummaries(data) {
        if (greenSummaryEl) {
          greenSummaryEl.innerText = formatSignalSummary("GREEN", data.signals?.green);
        }

        if (posSummaryEl) {
          posSummaryEl.innerText = formatSignalSummary("POS", data.signals?.pos);
        }

        if (chromSummaryEl) {
          chromSummaryEl.innerText = formatSignalSummary("CHROM", data.signals?.chrom);
        }
      }

      if (roiSamples.length < 20) {
        statusEl.innerText = "Collect at least 20 ROI samples before analysis.";
        return;
      }

      analyzeButton.disabled = true;
      analyzeButton.innerText = "Analyzing...";

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

        setCompactSignalSummaries(data);
        drawMainPulseWaveformFromAnalysis(data);

        const greenBpm = data.signals?.green?.spectral?.dominant_bpm;
        const posBpm = data.signals?.pos?.spectral?.dominant_bpm;
        const chromBpm = data.signals?.chrom?.spectral?.dominant_bpm;

        const greenSqi = data.signals?.green?.spectral?.sqi;
        const posSqi = data.signals?.pos?.spectral?.sqi;
        const chromSqi = data.signals?.chrom?.spectral?.sqi;

        const spectralConsensus = computeSpectralConsensusFromAnalysis(data);

        if (spectralConsensusEl) {
          spectralConsensusEl.innerText = formatBpm(spectralConsensus);
        }

        const compactSummary = [
          `status: ${data.status}`,
          `samples: ${data.sample_count}`,
          `duration_s: ${data.duration_s.toFixed(2)}`,
          `estimated_fps: ${data.estimated_fps.toFixed(2)}`,
          "",
          `spectral_consensus: ${formatBpm(spectralConsensus)}`,
          "",
          `GREEN: ${greenBpm === null || greenBpm === undefined ? "none" : greenBpm.toFixed(1)} bpm / SQI ${greenSqi === null || greenSqi === undefined ? "none" : greenSqi.toFixed(3)} / ${data.signals?.green?.spectral?.status ?? "unknown"}`,
          `POS:   ${posBpm === null || posBpm === undefined ? "none" : posBpm.toFixed(1)} bpm / SQI ${posSqi === null || posSqi === undefined ? "none" : posSqi.toFixed(3)} / ${data.signals?.pos?.spectral?.status ?? "unknown"}`,
          `CHROM: ${chromBpm === null || chromBpm === undefined ? "none" : chromBpm.toFixed(1)} bpm / SQI ${chromSqi === null || chromSqi === undefined ? "none" : chromSqi.toFixed(3)} / ${data.signals?.chrom?.spectral?.status ?? "unknown"}`,
          "",
          "Full response:",
          JSON.stringify(data, null, 2)
        ].join("\\n");

        outputEl.innerText = compactSummary;

        statusEl.innerText =
          `ROI series analyzed. Spectral consensus=${formatBpm(spectralConsensus)}. ` +
          `GREEN=${greenBpm?.toFixed(1) ?? "none"} BPM, ` +
          `POS=${posBpm?.toFixed(1) ?? "none"} BPM, ` +
          `CHROM=${chromBpm?.toFixed(1) ?? "none"} BPM.`;
      } catch (error) {
        outputEl.innerText = `ROI series analysis failed: ${error}`;
        statusEl.innerText = `ROI series analysis failed: ${error}`;

        drawMainPulseWaveformPlaceholder();

        if (greenSummaryEl) {
          greenSummaryEl.innerText = "Analysis failed";
        }

        if (posSummaryEl) {
          posSummaryEl.innerText = "Analysis failed";
        }

        if (chromSummaryEl) {
          chromSummaryEl.innerText = "Analysis failed";
        }
      } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerText = "Analyze ROI series";
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

      const lines = [
        `status: ${data.status}`,
        `model_hr_bpm: ${formatBpm(modelHr)}`,
        `spectral_consensus_bpm: ${formatBpm(consensus)}`,
        `model_minus_spectral: ${formatSignedBpm(difference)}`,
        "",
        `GREEN: ${formatBpm(green?.dominant_bpm)} / SQI ${green?.sqi?.toFixed(3) ?? "none"} / ${green?.status ?? "unknown"}`,
        `POS:   ${formatBpm(pos?.dominant_bpm)} / SQI ${pos?.sqi?.toFixed(3) ?? "none"} / ${pos?.status ?? "unknown"}`,
        `CHROM: ${formatBpm(chrom?.dominant_bpm)} / SQI ${chrom?.sqi?.toFixed(3) ?? "none"} / ${chrom?.status ?? "unknown"}`,
        "",
        `original_duration_s: ${windowMetadata.original_duration_s?.toFixed(2) ?? "unknown"}`,
        `used_duration_s: ${windowMetadata.used_duration_s?.toFixed(2) ?? "unknown"}`,
        `used_samples: ${windowMetadata.used_sample_count ?? "unknown"}`,
        `source_fps: ${data.model_input?.source_estimated_fps?.toFixed(2) ?? "unknown"}`,
        "",
        "Full response:",
        JSON.stringify(data, null, 2)
      ];

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
        text: lines.join("\\n")
      };
    }

    function ensureRepeatabilityTableExists(outputEl) {
      let container = document.getElementById("live-model-repeatability-container");

      if (container) {
        return container;
      }

      container = document.createElement("div");
      container.id = "live-model-repeatability-container";
      container.className = "mt-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm";

      container.innerHTML = `
        <div class="mb-2">
          <div class="text-sm font-semibold text-slate-900">Live prediction repeatability table</div>
          <div class="text-xs text-slate-600">
            Each row is one click of "Run live model prediction" using the current ROI sample buffer.
          </div>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full border-collapse text-xs">
            <thead>
              <tr class="border-b border-slate-200 text-left text-slate-500">
                <th class="py-2 pr-3">Run</th>
                <th class="py-2 pr-3">Model HR</th>
                <th class="py-2 pr-3">Spectral</th>
                <th class="py-2 pr-3">Diff</th>
                <th class="py-2 pr-3">GREEN SQI</th>
                <th class="py-2 pr-3">POS SQI</th>
                <th class="py-2 pr-3">CHROM SQI</th>
                <th class="py-2 pr-3">Used s</th>
                <th class="py-2 pr-3">Samples</th>
                <th class="py-2 pr-3">FPS</th>
              </tr>
            </thead>
            <tbody id="live-model-repeatability-table-body"></tbody>
          </table>
        </div>
      `;

      outputEl.insertAdjacentElement("afterend", container);

      return container;
    }

    function renderRepeatabilityTable(outputEl) {
      ensureRepeatabilityTableExists(outputEl);

      const tableBody = document.getElementById("live-model-repeatability-table-body");

      if (!tableBody) {
        return;
      }

      tableBody.innerHTML = "";

      for (const run of window.livePredictionRuns) {
        const row = document.createElement("tr");
        row.className = "border-b border-slate-100 text-slate-800";

        row.innerHTML = `
          <td class="py-2 pr-3">${run.runIndex}</td>
          <td class="py-2 pr-3 font-medium">${formatBpm(run.modelHr)}</td>
          <td class="py-2 pr-3">${formatBpm(run.consensus)}</td>
          <td class="py-2 pr-3">${formatSignedBpm(run.difference)}</td>
          <td class="py-2 pr-3">${formatNumber(run.greenSqi, 3)} / ${run.greenStatus}</td>
          <td class="py-2 pr-3">${formatNumber(run.posSqi, 3)} / ${run.posStatus}</td>
          <td class="py-2 pr-3">${formatNumber(run.chromSqi, 3)} / ${run.chromStatus}</td>
          <td class="py-2 pr-3">${formatNumber(run.usedDurationS, 2)}</td>
          <td class="py-2 pr-3">${run.usedSamples ?? "none"}</td>
          <td class="py-2 pr-3">${formatNumber(run.sourceFps, 2)}</td>
        `;

        tableBody.appendChild(row);
      }
    }

    function addPredictionRun(summary, outputEl) {
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

      renderRepeatabilityTable(outputEl);
    }

    async function runLiveModelPredictionInBackend() {
      const statusEl = document.getElementById("camera-status");
      const outputEl = document.getElementById("live-model-prediction-output");
      const runButton = document.getElementById("run-live-model-button");

      const modelHrEl = document.getElementById("live-model-hr-summary");
      const spectralConsensusEl = document.getElementById("spectral-consensus-summary");
      const modelDifferenceEl = document.getElementById("model-spectral-difference-summary");

      if (!window.livePredictionRuns) {
        window.livePredictionRuns = [];
      }

      if (roiSamples.length < 20) {
        statusEl.innerText = "Collect at least 20 ROI samples before live model prediction.";
        return;
      }

      runButton.disabled = true;
      runButton.innerText = "Predicting...";

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
        const summary = summarizeModelPrediction(data);

        if (modelHrEl) {
          modelHrEl.innerText = formatBpm(summary.modelHr);
        }

        if (spectralConsensusEl) {
          spectralConsensusEl.innerText = formatBpm(summary.consensus);
        }

        if (modelDifferenceEl) {
          modelDifferenceEl.innerText = formatSignedBpm(summary.difference);
        }

        outputEl.innerText = summary.text;

        addPredictionRun(summary, outputEl);

        statusEl.innerText =
          `Live model prediction completed. Model=${formatBpm(summary.modelHr)}, ` +
          `spectral consensus=${formatBpm(summary.consensus)}.`;
      } catch (error) {
        outputEl.innerText = `Live model prediction failed: ${error}`;
        statusEl.innerText = `Live model prediction failed: ${error}`;

        if (modelHrEl) {
          modelHrEl.innerText = "Prediction failed";
        }

        if (spectralConsensusEl) {
          spectralConsensusEl.innerText = "Prediction failed";
        }

        if (modelDifferenceEl) {
          modelDifferenceEl.innerText = "Prediction failed";
        }
      } finally {
        runButton.disabled = false;
        runButton.innerText = "Run live model prediction";
      }
    }

    document.addEventListener("DOMContentLoaded", () => {
      const refreshButton = document.getElementById("refresh-api-button");
      const startCameraButton = document.getElementById("start-camera-button");
      const captureFrameButton = document.getElementById("capture-frame-button");
      const sendFrameButton = document.getElementById("send-frame-button");
      const detectFaceButton = document.getElementById("detect-face-button");
      const stopCameraButton = document.getElementById("stop-camera-button");
      const startRoiSamplingButton = document.getElementById("start-roi-sampling-button");
      const stopRoiSamplingButton = document.getElementById("stop-roi-sampling-button");
      const clearRoiSamplesButton = document.getElementById("clear-roi-samples-button");
      const analyzeRoiSeriesButton = document.getElementById("analyze-roi-series-button");
      const runLiveModelButton = document.getElementById("run-live-model-button");

      drawAllRoiPlots();
      drawMainPulseWaveformPlaceholder();

      if (refreshButton) {
        refreshButton.addEventListener("click", refreshSyntheticResult);
      }

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
    """.strip()

    return Script(NotStr(script))