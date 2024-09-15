(() => {
  const apiBase = location.protocol === "file:" ? "http://localhost:5003" : location.origin;

  const els = {
    form: document.getElementById("analyzeForm"),
    fileDrop: document.getElementById("fileDrop"),
    fileInput: document.getElementById("fileInput"),
    fileTitle: document.getElementById("fileTitle"),
    fileSub: document.getElementById("fileSub"),
    detectorSelect: document.getElementById("detectorSelect"),
    classifyToggle: document.getElementById("classifyToggle"),
    nearFallsToggle: document.getElementById("nearFallsToggle"),
    analyzeButton: document.getElementById("analyzeButton"),
    resetButton: document.getElementById("resetButton"),
    formStatus: document.getElementById("formStatus"),
    healthStatus: document.getElementById("healthStatus"),
    refreshHealth: document.getElementById("refreshHealth"),
    resultMeta: document.getElementById("resultMeta"),
    emptyState: document.getElementById("emptyState"),
    resultsContent: document.getElementById("resultsContent"),
    fallBadge: document.getElementById("fallBadge"),
    confidenceValue: document.getElementById("confidenceValue"),
    confidenceBar: document.getElementById("confidenceBar"),
    activityValue: document.getElementById("activityValue"),
    characteristicsTags: document.getElementById("characteristicsTags"),
    metricsGrid: document.getElementById("metricsGrid"),
    pelvisChart: document.getElementById("pelvisChart"),
    headChart: document.getElementById("headChart"),
    velocityChart: document.getElementById("velocityChart"),
    tiltChart: document.getElementById("tiltChart"),
    fallTypeBody: document.getElementById("fallTypeBody"),
    nearFallsBody: document.getElementById("nearFallsBody"),
    rawJson: document.getElementById("rawJson"),
  };

  let lastTimeline = null;

  const metricConfig = [
    { key: "min_vertical_velocity_ms", label: "Min vertical velocity", unit: "m/s", digits: 2 },
    { key: "max_vertical_velocity_ms", label: "Max vertical velocity", unit: "m/s", digits: 2 },
    { key: "mean_speed_ms", label: "Mean COM speed", unit: "m/s", digits: 2 },
    { key: "height_drop_m", label: "Height drop", unit: "m", digits: 2 },
    { key: "height_drop_robust_m", label: "Robust height drop", unit: "m", digits: 2 },
    { key: "trunk_tilt_max_deg", label: "Max trunk tilt", unit: "deg", digits: 1 },
    { key: "trunk_tilt_final_deg", label: "Final trunk tilt", unit: "deg", digits: 1 },
    { key: "acc_max_g", label: "Max acceleration", unit: "g", digits: 2 },
    { key: "impact_z_score", label: "Impact z-score", unit: "", digits: 2 },
    { key: "descent_duration_s", label: "Descent duration", unit: "s", digits: 2 },
    { key: "mos_min_m", label: "MoS min", unit: "m", digits: 3 },
  ];

  function setStatus(kind, message) {
    els.formStatus.textContent = message;
    els.formStatus.classList.remove("good", "bad", "warn");
    if (kind) {
      els.formStatus.classList.add(kind);
    }
  }

  function setHealthStatus(kind, message) {
    els.healthStatus.textContent = message;
    els.healthStatus.classList.remove("good", "bad", "warn");
    if (kind) {
      els.healthStatus.classList.add(kind);
    }
  }

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "--";
    }
    return Number(value).toFixed(digits);
  }

  function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "--";
    }
    const numeric = Number(value);
    const percent = numeric <= 1 ? numeric * 100 : numeric;
    return `${percent.toFixed(1)}%`;
  }

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) {
      return "";
    }
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    const units = ["KB", "MB", "GB"];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return `${value.toFixed(1)} ${units[unitIndex]}`;
  }

  function humanizeKey(key) {
    return key
      .replace(/_/g, " ")
      .replace(/\bcom\b/gi, "COM")
      .replace(/\bmos\b/gi, "MoS");
  }

  function updateFileLabel() {
    const file = els.fileInput.files && els.fileInput.files[0];
    if (!file) {
      els.fileTitle.textContent = "Drop C3D file here";
      els.fileSub.textContent = "or click to browse";
      return;
    }
    els.fileTitle.textContent = file.name;
    els.fileSub.textContent = `${formatBytes(file.size)} file selected`;
  }

  function toggleResults(show) {
    els.resultsContent.classList.toggle("hidden", !show);
    els.emptyState.classList.toggle("hidden", show);
  }

  function clearResults() {
    toggleResults(false);
    els.resultMeta.textContent = "Awaiting analysis.";
    els.fallBadge.textContent = "Awaiting";
    els.fallBadge.classList.remove("good", "bad");
    els.confidenceValue.textContent = "--";
    els.confidenceBar.style.width = "0";
    els.activityValue.textContent = "--";
    els.characteristicsTags.innerHTML = "";
    els.metricsGrid.innerHTML = "";
    els.rawJson.textContent = "";
    els.fallTypeBody.textContent = "Not requested.";
    els.nearFallsBody.textContent = "Not requested.";
    lastTimeline = null;
    renderCharts(null);
  }

  function renderTags(container, tags) {
    container.innerHTML = "";
    if (!tags || tags.length === 0) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "None";
      container.appendChild(tag);
      return;
    }
    tags.forEach((text) => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = text;
      container.appendChild(tag);
    });
  }

  function renderMetrics(metrics) {
    els.metricsGrid.innerHTML = "";
    if (!metrics || Object.keys(metrics).length === 0) {
      const empty = document.createElement("div");
      empty.className = "metric-card";
      empty.textContent = "No metrics available.";
      els.metricsGrid.appendChild(empty);
      return;
    }

    const usedKeys = new Set();
    const cards = [];

    metricConfig.forEach((config) => {
      if (metrics[config.key] !== undefined) {
        usedKeys.add(config.key);
        cards.push({
          label: config.label,
          value: metrics[config.key],
          unit: config.unit,
          digits: config.digits,
        });
      }
    });

    Object.keys(metrics).forEach((key) => {
      if (usedKeys.has(key)) {
        return;
      }
      cards.push({
        label: humanizeKey(key),
        value: metrics[key],
        unit: "",
        digits: 2,
      });
    });

    cards.slice(0, 12).forEach((metric) => {
      const card = document.createElement("div");
      card.className = "metric-card";

      const label = document.createElement("div");
      label.className = "metric-label";
      label.textContent = metric.label;

      const value = document.createElement("div");
      value.className = "metric-value";
      const formatted = typeof metric.value === "number" ? formatNumber(metric.value, metric.digits) : metric.value;
      value.textContent = metric.unit ? `${formatted} ${metric.unit}` : `${formatted}`;

      card.appendChild(label);
      card.appendChild(value);
      els.metricsGrid.appendChild(card);
    });
  }

  function downsample(series, maxPoints) {
    if (!series || series.length <= maxPoints) {
      return series;
    }
    const sampled = [];
    const step = series.length / maxPoints;
    for (let i = 0; i < maxPoints; i += 1) {
      sampled.push(series[Math.floor(i * step)]);
    }
    return sampled;
  }

  function drawLineChart(canvas, data, options) {
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!data || data.length < 2) {
      ctx.fillStyle = "rgba(18, 33, 36, 0.45)";
      ctx.font = "12px sans-serif";
      ctx.fillText("No data", 12, rect.height / 2);
      return;
    }

    const series = downsample(data, 280);
    const min = Math.min(...series);
    const max = Math.max(...series);
    const range = max - min || 1;
    const pad = 12;
    const width = rect.width - pad * 2;
    const height = rect.height - pad * 2;

    ctx.strokeStyle = "rgba(18, 33, 36, 0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 2; i += 1) {
      const y = pad + (height / 2) * i;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + width, y);
      ctx.stroke();
    }

    if (options.zeroLine && min < 0 && max > 0) {
      const zeroY = pad + ((max - 0) / range) * height;
      ctx.strokeStyle = "rgba(31, 58, 95, 0.3)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad, zeroY);
      ctx.lineTo(pad + width, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = options.lineColor;
    ctx.lineWidth = 2.2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    series.forEach((value, index) => {
      const x = pad + (index / (series.length - 1)) * width;
      const y = pad + ((max - value) / range) * height;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    if (options.fillColor) {
      ctx.lineTo(pad + width, pad + height);
      ctx.lineTo(pad, pad + height);
      ctx.closePath();
      ctx.fillStyle = options.fillColor;
      ctx.fill();
    }
  }

  function renderCharts(timeline) {
    lastTimeline = timeline;
    drawLineChart(els.pelvisChart, timeline ? timeline.pelvis_height : null, {
      lineColor: "#e07a25",
      fillColor: "rgba(224, 122, 37, 0.08)",
    });
    drawLineChart(els.headChart, timeline ? timeline.head_height : null, {
      lineColor: "#2a9d8f",
      fillColor: "rgba(42, 157, 143, 0.1)",
    });
    drawLineChart(els.velocityChart, timeline ? timeline.vertical_velocity : null, {
      lineColor: "#1f3a5f",
      fillColor: "rgba(31, 58, 95, 0.08)",
      zeroLine: true,
    });
    drawLineChart(els.tiltChart, timeline ? timeline.trunk_tilt_deg : null, {
      lineColor: "#8c3d0c",
      fillColor: "rgba(140, 61, 12, 0.08)",
    });
  }

  function renderAnalysis(result) {
    toggleResults(true);
    els.rawJson.textContent = JSON.stringify(result, null, 2);

    const fallDetected = Boolean(result.fall_detected);
    els.fallBadge.textContent = fallDetected ? "Fall detected" : "No fall detected";
    els.fallBadge.classList.toggle("bad", fallDetected);
    els.fallBadge.classList.toggle("good", !fallDetected);

    const confidence = Number(result.confidence);
    const confidencePercent = Math.max(0, Math.min(100, confidence <= 1 ? confidence * 100 : confidence));
    els.confidenceValue.textContent = formatPercent(confidence);
    els.confidenceBar.style.width = `${confidencePercent}%`;

    els.activityValue.textContent = result.activity_type || "--";
    renderTags(els.characteristicsTags, result.characteristics);

    renderMetrics(result.metrics);
    renderCharts(result.timeline_data);

    const timeline = result.timeline_data || {};
    const meta = [];
    if (timeline.frame_rate) {
      meta.push(`${formatNumber(timeline.frame_rate, 1)} fps`);
    }
    if (timeline.n_frames) {
      meta.push(`${timeline.n_frames} frames`);
    }
    if (timeline.impact_occurred) {
      meta.push(`impact at ${formatNumber(timeline.impact_time_s, 2)} s`);
    }
    els.resultMeta.textContent = meta.length ? meta.join(" | ") : "Analysis complete.";
  }

  function renderFallType(data) {
    els.fallTypeBody.innerHTML = "";
    if (!data) {
      els.fallTypeBody.textContent = "Not requested.";
      return;
    }
    if (data.error) {
      els.fallTypeBody.textContent = data.error;
      return;
    }
    const value = document.createElement("div");
    value.className = "extra-value";
    value.textContent = data.fall_type ? data.fall_type.replace(/_/g, " ") : "Unknown";
    els.fallTypeBody.appendChild(value);

    const confidence = document.createElement("div");
    confidence.className = "extra-sub";
    confidence.textContent = `Confidence: ${formatPercent(data.confidence)}`;
    els.fallTypeBody.appendChild(confidence);

    if (data.evidence && data.evidence.length) {
      const tags = document.createElement("div");
      tags.className = "tag-row";
      data.evidence.forEach((item) => {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = item;
        tags.appendChild(tag);
      });
      els.fallTypeBody.appendChild(tags);
    }

    const details = [];
    if (data.horizontal_vertical_ratio !== undefined) {
      details.push(`HV ratio: ${formatNumber(data.horizontal_vertical_ratio, 2)}`);
    }
    if (data.horizontal_velocity_direction) {
      details.push(`Velocity dir: ${data.horizontal_velocity_direction.join(", ")}`);
    }
    if (data.trunk_rotation_direction) {
      details.push(`Trunk rotation: ${data.trunk_rotation_direction}`);
    }
    if (details.length) {
      const extra = document.createElement("div");
      extra.className = "extra-sub";
      extra.textContent = details.join(" | ");
      els.fallTypeBody.appendChild(extra);
    }
  }

  function renderNearFalls(data) {
    els.nearFallsBody.innerHTML = "";
    if (!data) {
      els.nearFallsBody.textContent = "Not requested.";
      return;
    }
    if (data.error) {
      els.nearFallsBody.textContent = data.error;
      return;
    }
    if (!data.events || data.events.length === 0) {
      els.nearFallsBody.textContent = "No near-falls detected.";
      return;
    }
    const count = document.createElement("div");
    count.className = "extra-value";
    count.textContent = `${data.count} events`;
    els.nearFallsBody.appendChild(count);

    const list = document.createElement("div");
    list.className = "extra-list";
    data.events.slice(0, 4).forEach((event, index) => {
      const item = document.createElement("div");
      item.className = "extra-item";
      item.textContent = `Event ${index + 1}: ${formatNumber(event.time_start_s, 2)}s to ${formatNumber(event.time_end_s, 2)}s, severity ${formatNumber(event.severity, 2)}`;
      list.appendChild(item);
    });
    els.nearFallsBody.appendChild(list);

    if (data.events.length > 4) {
      const note = document.createElement("div");
      note.className = "extra-sub";
      note.textContent = "Showing first 4 events.";
      els.nearFallsBody.appendChild(note);
    }
  }

  function makeFormData(file) {
    const formData = new FormData();
    formData.append("file", file);
    return formData;
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";
    let payload = {};
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      const text = await response.text();
      payload = text ? { error: text } : {};
    }
    if (!response.ok) {
      const message = payload.error || response.statusText || "Request failed";
      throw new Error(message);
    }
    return payload;
  }

  async function updateHealth() {
    setHealthStatus("warn", "Checking API...");
    try {
      const data = await fetchJson(`${apiBase}/api/v2/health`, { method: "GET" });
      const detectors = data.detectors || {};
      const parts = [
        `Status: ${data.status || "unknown"}`,
        `Rules: ${detectors.rules ? "ready" : "off"}`,
        `LSTM: ${detectors.lstm ? "ready" : "off"}`,
      ];
      setHealthStatus("good", parts.join(" | "));
    } catch (err) {
      setHealthStatus("bad", `API unavailable: ${err.message}`);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const file = els.fileInput.files && els.fileInput.files[0];
    if (!file) {
      setStatus("bad", "Select a .c3d file first.");
      return;
    }

    els.analyzeButton.disabled = true;
    els.resetButton.disabled = true;
    setStatus("warn", "Running analysis...");
    els.fallTypeBody.textContent = els.classifyToggle.checked ? "Running..." : "Not requested.";
    els.nearFallsBody.textContent = els.nearFallsToggle.checked ? "Running..." : "Not requested.";

    try {
      const detector = els.detectorSelect.value;
      const url =
        detector === "lstm"
          ? `${apiBase}/api/v2/analyze?detector=lstm`
          : `${apiBase}/api/v2/analyze`;
      const analysis = await fetchJson(url, {
        method: "POST",
        body: makeFormData(file),
      });
      renderAnalysis(analysis);
      setStatus("good", "Analysis complete.");

      const extraTasks = [];
      if (els.classifyToggle.checked) {
        extraTasks.push(
          fetchJson(`${apiBase}/api/v2/classify-fall-type`, {
            method: "POST",
            body: makeFormData(file),
          })
            .then((data) => renderFallType(data))
            .catch((err) => renderFallType({ error: err.message }))
        );
      } else {
        renderFallType(null);
      }

      if (els.nearFallsToggle.checked) {
        extraTasks.push(
          fetchJson(`${apiBase}/api/v2/detect-near-falls`, {
            method: "POST",
            body: makeFormData(file),
          })
            .then((data) => renderNearFalls(data))
            .catch((err) => renderNearFalls({ error: err.message }))
        );
      } else {
        renderNearFalls(null);
      }

      if (extraTasks.length) {
        await Promise.all(extraTasks);
      }
    } catch (err) {
      setStatus("bad", err.message);
      toggleResults(false);
    } finally {
      els.analyzeButton.disabled = false;
      els.resetButton.disabled = false;
    }
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    els.fileDrop.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.fileDrop.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    els.fileDrop.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.fileDrop.classList.remove("dragover");
      if (eventName === "drop" && event.dataTransfer && event.dataTransfer.files.length) {
        try {
          els.fileInput.files = event.dataTransfer.files;
        } catch (err) {
          return;
        }
        if (els.fileInput.files && els.fileInput.files.length) {
          updateFileLabel();
        }
      }
    });
  });

  els.fileInput.addEventListener("change", updateFileLabel);
  els.form.addEventListener("submit", handleSubmit);
  els.resetButton.addEventListener("click", () => {
    els.form.reset();
    updateFileLabel();
    clearResults();
    setStatus("", "Ready.");
  });
  els.refreshHealth.addEventListener("click", updateHealth);
  window.addEventListener("resize", () => {
    if (lastTimeline) {
      renderCharts(lastTimeline);
    }
  });

  updateFileLabel();
  clearResults();
  updateHealth();
})();
