const SVG_NS = "http://www.w3.org/2000/svg";

const fallbackMetrics = {
  final_step: 19073,
  total_tokens_processed: 9999745024,
  train_loss: 3.1033270359039307,
  validation_loss: 3.0308315753936768,
  validation_perplexity: 20.71445105696278,
  hellaswag_results: {
    accuracy_percent: 30.033857797251545,
    correct: 3016,
    total: 10042,
  },
  training_time_hours: 9.401894711041386,
  throughput_tokens_per_second: 295441.188875369,
  estimated_cost_usd: 25.385115719811744,
};

const format = {
  integer: new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }),
  decimal2: new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  decimal4: new Intl.NumberFormat("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 }),
};

function setText(selector, value) {
  document.querySelectorAll(selector).forEach((element) => {
    element.textContent = value;
  });
}

function hydrateMetrics(metrics) {
  setText('[data-metric="parameters"]', "124.48M");
  setText('[data-metric="tokens"]', `${(metrics.total_tokens_processed / 1e9).toFixed(2)}B`);
  setText('[data-metric="tokens-exact"]', `${format.integer.format(metrics.total_tokens_processed)} tokens`);
  setText('[data-metric="steps"]', `${format.integer.format(metrics.final_step)} optimizer steps`);
  setText('[data-metric="train-loss"]', format.decimal4.format(metrics.train_loss));
  setText('[data-metric="validation-loss"]', format.decimal4.format(metrics.validation_loss));
  setText('[data-metric="perplexity"]', format.decimal2.format(metrics.validation_perplexity));
  setText('[data-metric="hellaswag"]', `${metrics.hellaswag_results.accuracy_percent.toFixed(2)}%`);
  setText('[data-metric="hellaswag-long"]', `${metrics.hellaswag_results.accuracy_percent.toFixed(4)}%`);
  setText('[data-metric="hours"]', `${metrics.training_time_hours.toFixed(2)} h`);
  setText('[data-metric="throughput-short"]', `${Math.round(metrics.throughput_tokens_per_second / 1000)}k`);
  setText('[data-metric="cost"]', `$${metrics.estimated_cost_usd.toFixed(2)}`);
}

function parseHistory(csvText) {
  const rows = { train: [], validation: [], hellaswag: [] };
  const lines = csvText.split(/\r?\n/).slice(1);

  for (const line of lines) {
    const columns = line.split(",");
    const event = columns[0];
    if (!Object.hasOwn(rows, event)) continue;

    const step = Number(columns[1]);
    if (!Number.isFinite(step)) continue;

    if (event === "train") {
      const loss = Number(columns[2]);
      const learningRate = Number(columns[3]);
      const throughput = Number(columns[6]);
      if (Number.isFinite(loss) && Number.isFinite(learningRate) && Number.isFinite(throughput)) {
        rows.train.push({ step, loss, learningRate, throughput });
      }
    } else if (event === "validation") {
      const loss = Number(columns[10]);
      if (Number.isFinite(loss)) rows.validation.push({ step, loss });
    } else {
      const accuracy = Number(columns[11]);
      if (Number.isFinite(accuracy)) rows.hellaswag.push({ step, accuracy });
    }
  }

  return rows;
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function extent(values) {
  return [Math.min(...values), Math.max(...values)];
}

function paddedExtent(values, paddingRatio = 0.08) {
  const [minimum, maximum] = extent(values);
  const span = maximum - minimum || Math.abs(maximum) || 1;
  return [minimum - span * paddingRatio, maximum + span * paddingRatio];
}

function rollingMean(data, accessor, windowSize = 18) {
  return data.map((point, index) => {
    const start = Math.max(0, index - windowSize + 1);
    const window = data.slice(start, index + 1);
    return {
      ...point,
      rolling: window.reduce((sum, item) => sum + accessor(item), 0) / window.length,
    };
  });
}

function linePath(data, xScale, yScale, accessor) {
  return data
    .map((point, index) => `${index === 0 ? "M" : "L"}${xScale(point.step).toFixed(2)},${yScale(accessor(point)).toFixed(2)}`)
    .join(" ");
}

function nearestPoint(data, step) {
  let low = 0;
  let high = data.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (data[middle].step < step) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return data[0];
  const before = data[low - 1];
  const after = data[low];
  return Math.abs(before.step - step) <= Math.abs(after.step - step) ? before : after;
}

function renderLineChart(container, options) {
  const data = options.data;
  if (!data.length) {
    container.innerHTML = '<div class="chart-error">The committed history could not be loaded. Open this page through a local web server or GitHub Pages to view the chart.</div>';
    return;
  }

  const width = Math.max(320, Math.round(container.clientWidth || 900));
  const height = width < 520 ? 280 : 370;
  const margin = {
    top: 30,
    right: width < 520 ? 14 : 26,
    bottom: 48,
    left: width < 520 ? 48 : 66,
  };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const [xMinimum, xMaximum] = extent(data.map((point) => point.step));
  const [yMinimum, yMaximum] = options.yDomain || paddedExtent(data.map(options.accessor), options.padding ?? 0.08);
  const xScale = (value) => margin.left + ((value - xMinimum) / (xMaximum - xMinimum || 1)) * plotWidth;
  const yScale = (value) => margin.top + (1 - (value - yMinimum) / (yMaximum - yMinimum || 1)) * plotHeight;
  const xInvert = (value) => xMinimum + ((value - margin.left) / plotWidth) * (xMaximum - xMinimum);

  container.innerHTML = "";
  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
    "aria-label": options.ariaLabel,
  });
  const title = svgElement("title");
  title.textContent = options.ariaLabel;
  svg.append(title);

  const yTicks = 5;
  for (let index = 0; index < yTicks; index += 1) {
    const ratio = index / (yTicks - 1);
    const y = margin.top + ratio * plotHeight;
    const value = yMaximum - ratio * (yMaximum - yMinimum);
    svg.append(svgElement("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "grid-line" }));
    const label = svgElement("text", { x: margin.left - 10, y: y + 3, "text-anchor": "end", class: "axis-label" });
    label.textContent = options.yFormat(value);
    svg.append(label);
  }

  const xTicks = width < 520 ? 3 : 5;
  for (let index = 0; index < xTicks; index += 1) {
    const ratio = index / (xTicks - 1);
    const x = margin.left + ratio * plotWidth;
    const value = xMinimum + ratio * (xMaximum - xMinimum);
    const label = svgElement("text", { x, y: height - 18, "text-anchor": "middle", class: "axis-label" });
    label.textContent = format.integer.format(Math.round(value));
    svg.append(label);
  }

  const xTitle = svgElement("text", {
    x: width - margin.right,
    y: height - 4,
    "text-anchor": "end",
    class: "axis-label",
  });
  xTitle.textContent = "optimizer step";
  svg.append(xTitle);

  if (options.rawData) {
    const raw = svgElement("path", {
      d: linePath(options.rawData, xScale, yScale, options.rawAccessor),
      class: "raw-line",
    });
    svg.append(raw);
  }

  const path = svgElement("path", {
    d: linePath(data, xScale, yScale, options.accessor),
    class: "main-line",
  });
  svg.append(path);

  if (options.showPoints) {
    data.forEach((point) => {
      svg.append(svgElement("circle", {
        cx: xScale(point.step),
        cy: yScale(options.accessor(point)),
        r: width < 520 ? 2.2 : 3.2,
        class: "point",
      }));
    });
  }

  const hoverLine = svgElement("line", {
    x1: margin.left,
    y1: margin.top,
    x2: margin.left,
    y2: margin.top + plotHeight,
    class: "hover-line",
  });
  const hoverDot = svgElement("circle", { cx: 0, cy: 0, r: 4.5, class: "hover-dot" });
  svg.append(hoverLine, hoverDot);

  const overlay = svgElement("rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    fill: "transparent",
  });
  svg.append(overlay);

  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  container.append(svg, tooltip);

  overlay.addEventListener("pointermove", (event) => {
    const bounds = svg.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / bounds.width) * width;
    const point = nearestPoint(data, xInvert(Math.max(margin.left, Math.min(width - margin.right, viewX))));
    const x = xScale(point.step);
    const y = yScale(options.accessor(point));
    hoverLine.setAttribute("x1", x);
    hoverLine.setAttribute("x2", x);
    hoverLine.style.opacity = "1";
    hoverDot.setAttribute("cx", x);
    hoverDot.setAttribute("cy", y);
    hoverDot.style.opacity = "1";

    const displayX = (x / width) * bounds.width;
    const displayY = (y / height) * bounds.height;
    tooltip.style.left = `${displayX}px`;
    tooltip.style.top = `${displayY}px`;
    tooltip.style.opacity = "1";
    tooltip.innerHTML = `<strong>${options.tooltipFormat(options.accessor(point))}</strong><span>step ${format.integer.format(point.step)}</span>`;
  });

  overlay.addEventListener("pointerleave", () => {
    hoverLine.style.opacity = "0";
    hoverDot.style.opacity = "0";
    tooltip.style.opacity = "0";
  });
}

function renderCharts(history) {
  const rollingLoss = rollingMean(history.train, (point) => point.loss, 18);

  renderLineChart(document.querySelector("#training-loss-chart"), {
    data: rollingLoss,
    rawData: history.train,
    accessor: (point) => point.rolling,
    rawAccessor: (point) => point.loss,
    yDomain: paddedExtent(history.train.map((point) => point.loss), 0.04),
    yFormat: (value) => value.toFixed(2),
    tooltipFormat: (value) => `rolling loss ${value.toFixed(4)}`,
    ariaLabel: "Training loss from optimizer step 13,140 to 19,070, showing raw values and an 18-point rolling mean.",
  });

  renderLineChart(document.querySelector("#validation-loss-chart"), {
    data: history.validation,
    accessor: (point) => point.loss,
    yFormat: (value) => value.toFixed(3),
    tooltipFormat: (value) => `validation loss ${value.toFixed(4)}`,
    showPoints: true,
    ariaLabel: "Validation loss across 25 evaluations from optimizer step 13,250 through the terminal checkpoint at step 19,073.",
  });

  renderLineChart(document.querySelector("#learning-rate-chart"), {
    data: history.train,
    accessor: (point) => point.learningRate,
    yDomain: [0.000055, 0.000198],
    yFormat: (value) => value.toExponential(1),
    tooltipFormat: (value) => `learning rate ${value.toExponential(3)}`,
    ariaLabel: "Cosine-decayed learning rate over the committed final continuation of training.",
  });

  renderLineChart(document.querySelector("#throughput-chart"), {
    data: history.train,
    accessor: (point) => point.throughput,
    yFormat: (value) => `${Math.round(value / 1000)}k`,
    tooltipFormat: (value) => `${format.integer.format(value)} tokens/s`,
    ariaLabel: "Tokens processed per second over the committed final continuation of single-H100 training.",
  });
}

function setupArchitectureInspector() {
  const activeLabel = document.querySelector("[data-active-layer]");
  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-layer]").forEach((candidate) => {
        candidate.classList.remove("is-selected");
        candidate.setAttribute("aria-pressed", "false");
      });
      button.classList.add("is-selected");
      button.setAttribute("aria-pressed", "true");
      activeLabel.textContent = String(button.dataset.layer).padStart(2, "0");
    });
  });
}

function setupSamples() {
  const tabs = [...document.querySelectorAll("[data-sample]")];
  const panels = [...document.querySelectorAll("[data-sample-panel]")];

  function activate(tab) {
    tabs.forEach((candidate) => {
      candidate.setAttribute("aria-selected", String(candidate === tab));
      candidate.tabIndex = candidate === tab ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.samplePanel !== tab.dataset.sample;
    });
    tab.focus({ preventScroll: true });
  }

  tabs.forEach((tab, index) => {
    tab.tabIndex = index === 0 ? 0 : -1;
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let targetIndex = index;
      if (event.key === "ArrowLeft") targetIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") targetIndex = (index + 1) % tabs.length;
      if (event.key === "Home") targetIndex = 0;
      if (event.key === "End") targetIndex = tabs.length - 1;
      activate(tabs[targetIndex]);
    });
  });
}

function setupScrollState() {
  const progress = document.querySelector(".reading-progress span");
  const railLinks = [...document.querySelectorAll(".section-rail a")];
  const sections = [...document.querySelectorAll("[data-section]")];

  const updateProgress = () => {
    const scrollRange = document.documentElement.scrollHeight - window.innerHeight;
    const percentage = scrollRange > 0 ? (window.scrollY / scrollRange) * 100 : 0;
    progress.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
  };
  updateProgress();
  window.addEventListener("scroll", updateProgress, { passive: true });

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      railLinks.forEach((link) => link.classList.toggle("is-active", link.hash === `#${visible.target.id}`));
    },
    { rootMargin: "-18% 0px -66%", threshold: [0, 0.1, 0.4] },
  );
  sections.forEach((section) => observer.observe(section));
}

function setupCopyButton() {
  const button = document.querySelector("[data-copy]");
  if (!button) return;
  button.addEventListener("click", async () => {
    const previous = button.textContent;
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select command";
    }
    window.setTimeout(() => {
      button.textContent = previous;
    }, 1600);
  });
}

async function loadEvidence() {
  let metrics = window.MODEL_CARD_METRICS || fallbackMetrics;
  let history = window.MODEL_CARD_HISTORY || null;

  try {
    if (history) throw new Error("Embedded evidence available");
    const [metricsResponse, historyResponse] = await Promise.all([
      fetch("./data/final_metrics.json"),
      fetch("./data/training_history.csv"),
    ]);
    if (metricsResponse.ok) metrics = await metricsResponse.json();
    if (historyResponse.ok) history = parseHistory(await historyResponse.text());
  } catch (error) {
    if (!history) console.warn("Evidence files are unavailable in this browsing context.", error);
  }

  hydrateMetrics(metrics);
  if (!history) {
    document.querySelectorAll(".chart").forEach((container) => {
      container.innerHTML = '<div class="chart-error">Charts load from the repository’s committed CSV when this page is served over HTTP.</div>';
    });
    return;
  }

  renderCharts(history);
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => renderCharts(history), 180);
  });
}

setupArchitectureInspector();
setupSamples();
setupScrollState();
setupCopyButton();
loadEvidence();
