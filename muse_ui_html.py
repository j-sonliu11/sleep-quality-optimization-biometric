# muse_modular/muse_ui_html.py
from __future__ import annotations

from muse_config import DARK_BG, DARK_PLOT, DARK_GRID, DARK_TEXT

# NOTE: This is your exact html_template from the big muse.py, moved into its own module.
html_template = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {
      margin:0; padding:0;
      background:__DARK_BG__;
      color:__DARK_TEXT__;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    .grid { display:grid; grid-template-columns:1fr; gap:40px; }

    .card {
      background:__DARK_BG__;
      border-radius:12px;
      overflow: visible;
      padding-top: 6px;
      padding-bottom: 8px;
    }
    #ppgcard { overflow: visible; position: relative; z-index: 20; }

    .ppghead{
      display:flex; align-items:center; justify-content:space-between;
      padding: 10px 12px 10px 12px;
      gap: 14px;
      flex-wrap: wrap;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }
    .ppghead .left { font-size: 1.05rem; opacity:0.95; }
    .ppghead .right{
      font-size: 0.95rem;
      opacity:0.88;
      display:flex;
      gap:14px;
      align-items:baseline;
      flex-wrap:wrap;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }
    .pill{
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.92rem;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.06);
      white-space: nowrap;
    }
    .muted { opacity:0.85; }

    .charthead{
      padding: 12px 12px 4px 12px;
      font-size: 1.02rem;
      opacity: 0.95;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }

    .tip {
      position: relative;
      display: inline-flex;
      align-items: baseline;
      gap: 6px;
      cursor: help;
      border-bottom: 1px dotted rgba(255,255,255,0.35);
      z-index: 3000;
      overflow: visible;
    }
    .tip .bubble {
      position: absolute;
      left: 0;
      top: 130%;
      min-width: 220px;
      max-width: 320px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(17,24,39,0.94);
      border: 1px solid rgba(255,255,255,0.14);
      box-shadow: 0 10px 22px rgba(0,0,0,0.35);
      color: rgba(255,255,255,0.92);
      font-size: 0.92rem;
      line-height: 1.25rem;
      z-index: 4000;
      opacity: 0;
      transform: translateY(-4px);
      pointer-events: none;
      transition: opacity 120ms ease, transform 120ms ease;
    }
    .tip:hover .bubble {
      opacity: 1;
      transform: translateY(0px);
    }

    .tip.tip-rmssd .bubble {
      left: auto !important;
      right: 0 !important;
    }

.dc-panel-hook { display: none; }

div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stMainBlockContainer"],
div[data-testid="stBlock"],
div[data-testid="stElementContainer"],
div.block-container {
  overflow: visible !important;
}

div[data-testid="column"]:has(.dc-panel-hook) {
  position: sticky !important;
  top: 0.85rem !important;
  align-self: flex-start !important;
  z-index: 30 !important;
}

div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.9rem 0.9rem 0.9rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.22);

  max-height: calc(100vh - 1.7rem);
  overflow: auto !important;
}

div[data-testid="column"]:has(.dc-panel-hook) .element-container:first-of-type {
  margin-bottom: 0.2rem;
}

@media (max-width: 899px) {
  div[data-testid="column"]:has(.dc-panel-hook) {
    position: static !important;
    top: auto !important;
    z-index: auto !important;
  }

  div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
    max-height: none !important;
    overflow: visible !important;
  }
}

@media (max-width: 899px) {
  div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
    position: static;
    top: auto;
    max-height: none;
    overflow: visible;
  }
}
  </style>
</head>
<body>
  <div class="grid">
    <div class="card" id="eegcard" style="display:none;"><div id="eeg" style="height:440px;"></div></div>

    <div class="card" id="ppgcard" style="display:none;">
      <div class="ppghead">
        <div class="left">
          ❤️ <b>Heart Rate</b>
          <span class="muted">(inst / smooth):</span>
          <b><span id="hrInst">—</span></b> / <b><span id="hrSmooth">—</span></b>
          <span class="muted">BPM</span>
        </div>
        <div class="right">
          <span class="pill" id="qualPill">PPG: —</span>

          <span class="tip tip-rmssd">
            <span class="muted">RMSSD</span>
            <span class="bubble">
              <b>RMSSD</b> is an HRV metric: it measures short-term beat-to-beat variability
              (root mean square of successive IBI differences). Higher often means more
              parasympathetic “recovery”, but only if the PPG signal is clean.
            </span>
          </span>
          <b><span id="rmssd">—</span></b><span class="muted"> ms</span>

          <span class="muted" id="hint"></span>
        </div>
      </div>

      <div id="ppg" style="height:300px;"></div>

      <div class="charthead">
        <span class="tip">
          <span><b>IBI trend</b> <span class="muted">(ms)</span></span>
          <span class="bubble">
            <b>IBI</b> (inter-beat interval) is the time between detected beats.
            Stable, smooth IBI points usually mean good peak detection. Noisy spikes
            often mean motion/contact issues.
          </span>
        </span>
      </div>
      <div id="ibi" style="height:220px;"></div>
    </div>

    <div class="card" id="bandscard" style="display:none;"><div id="bands" style="height:320px;"></div></div>
    <div class="card" id="psdcard" style="display:none;"><div id="psd" style="height:360px;"></div></div>
  </div>

<script>
const HOST = "127.0.0.1";
const PORT = __PORT__;
let lastUpdateMs = 200;

function darkLayoutBase(title) {
  return {
    title: {text: title},
    paper_bgcolor: "__DARK_BG__",
    plot_bgcolor: "__DARK_PLOT__",
    font: {color: "__DARK_TEXT__"},
    margin: {l:80,r:25,t:45,b:55},
    xaxis: { gridcolor: "__DARK_GRID__", zerolinecolor: "__DARK_GRID__", linecolor: "__DARK_GRID__", tickcolor: "__DARK_GRID__" },
    yaxis: { gridcolor: "__DARK_GRID__", zerolinecolor: "__DARK_GRID__", linecolor: "__DARK_GRID__", tickcolor: "__DARK_GRID__" },
  };
}

function robustZ(arr) {
  const x = arr.map(v => (Number.isFinite(v) ? v : NaN));
  const finite = x.filter(v => Number.isFinite(v));
  if (finite.length < 10) return x.map(_ => 0);

  finite.sort((a,b) => a-b);
  const mid = Math.floor(finite.length/2);
  const med = (finite.length % 2) ? finite[mid] : 0.5*(finite[mid-1]+finite[mid]);

  const absdev = finite.map(v => Math.abs(v - med)).sort((a,b)=>a-b);
  const mad = (absdev.length % 2) ? absdev[mid] : 0.5*(absdev[mid-1]+absdev[mid]);
  const scale = (mad > 1e-9) ? mad : 1.0;

  return x.map(v => {
    if (!Number.isFinite(v)) return 0;
    let z = (v - med) / scale;
    if (z > 8) z = 8;
    if (z < -8) z = -8;
    return z;
  });
}

function setQualityPill(label, score) {
  const pill = document.getElementById("qualPill");
  let bg = "rgba(255,255,255,0.06)";
  let bd = "rgba(255,255,255,0.16)";
  if (label === "Good") {
    bg = "rgba(34,197,94,0.18)";
    bd = "rgba(34,197,94,0.45)";
  } else if (label === "OK") {
    bg = "rgba(234,179,8,0.18)";
    bd = "rgba(234,179,8,0.45)";
  } else {
    bg = "rgba(239,68,68,0.16)";
    bd = "rgba(239,68,68,0.45)";
  }
  pill.style.background = bg;
  pill.style.borderColor = bd;
  pill.textContent = `PPG: ${label} (${Math.round(100*score)}%)`;
}

async function pollOnce() {
  const url = "http://" + HOST + ":" + PORT + "/data";
  try {
    const resp = await fetch(url, {cache:"no-store"});
    const data = await resp.json();
    if (!data.ok) return;

    if (data.update_ms && Math.abs(data.update_ms - lastUpdateMs) > 1) {
      lastUpdateMs = data.update_ms;
      clearInterval(window.__timer);
      window.__timer = setInterval(pollOnce, lastUpdateMs);
    }

    const hasEEG = (data.t && data.t.length && data.y && data.y.length);
    const hasPPG = (data.ppg_t && data.ppg_t.length && data.ppg_y && data.ppg_y.length && data.ppg_channels && data.ppg_channels.length);
    const hasBands = (data.bp_frac && data.bp_frac.length && data.bands && data.bands.length && data.channels && data.channels.length);
    const hasPSD = (data.f && data.f.length && data.psd_db && data.psd_db.length);

    document.getElementById("eegcard").style.display = hasEEG ? "block" : "none";
    document.getElementById("ppgcard").style.display = hasPPG ? "block" : "none";
    document.getElementById("bandscard").style.display = (hasEEG && hasBands) ? "block" : "none";
    document.getElementById("psdcard").style.display = (hasEEG && hasPSD) ? "block" : "none";

    const isPaused = !!data.paused;
    document.getElementById("hint").innerText = (hasPPG || hasEEG) ? (isPaused ? "Paused — showing last captured data." : "") : "";

    document.getElementById("hrInst").innerText =
      (data.hr_bpm_inst === null || data.hr_bpm_inst === undefined) ? "—" : Math.round(data.hr_bpm_inst).toString();
    document.getElementById("hrSmooth").innerText =
      (data.hr_bpm_smooth === null || data.hr_bpm_smooth === undefined) ? "—" : Math.round(data.hr_bpm_smooth).toString();
    document.getElementById("rmssd").innerText =
      (data.rmssd_ms === null || data.rmssd_ms === undefined) ? "—" : Math.round(data.rmssd_ms).toString();
    setQualityPill(data.ppg_quality || "Bad", data.ppg_quality_score || 0);

    if (hasEEG) {
      const traces = data.channels.map((ch, i) => ({
        x: data.t, y: data.y[i], mode:"lines", name: ch,
        line: {width: 1.6},
      }));

      const anns = data.channels.map((ch, i) => ({
        x: data.t[0], xref: "x",
        y: data.base[i], yref: "y",
        text: ch,
        showarrow: false,
        xanchor: "right",
        font: {size: 12, color: "__DARK_TEXT__"},
      }));

      const eegLayout = darkLayoutBase("EEG (rolling / seismograph style)");
      eegLayout.showlegend = false;
      eegLayout.xaxis.title = "Time (s) (now = 0)";
      eegLayout.yaxis.title = "Amplitude (µV)  (offset display)";
      eegLayout.annotations = anns;
      Plotly.react("eeg", traces, eegLayout, {displayModeBar:false, responsive:true});
    }

    if (hasPPG) {
      const ppgTraces = data.ppg_channels.map((ch, i) => {
        const z = robustZ(data.ppg_y[i]);
        return {
          x: data.ppg_t,
          y: z,
          mode:"lines",
          name: ch + " (norm)",
          line: {width: 1.5},
        };
      });

      const bandOK = (data.ppg_band_t && data.ppg_band_y && data.ppg_band_t.length && data.ppg_band_y.length);
      if (bandOK) {
        ppgTraces.unshift({
          x: data.ppg_band_t,
          y: data.ppg_band_y,
          mode: "lines",
          name: "Bandpassed (HR)",
          line: {width: 2.2},
          opacity: 0.95,
        });

        if (data.ppg_peak_t && data.ppg_peak_t.length) {
          ppgTraces.unshift({
            x: data.ppg_peak_t,
            y: data.ppg_peak_t.map(_ => 0),
            mode: "markers",
            name: "Peaks",
            marker: {size: 7, symbol: "circle"},
            opacity: 0.95
          });
        }
      }

      const ppgLayout = darkLayoutBase("PPG (rolling): normalized raw + bandpassed + peaks");
      ppgLayout.xaxis.title = "Time (s) (now = 0)";
      ppgLayout.yaxis.title = "Robust z-score (a.u.)";
      ppgLayout.showlegend = true;
      ppgLayout.margin = {l:80,r:25,t:45,b:55};
      Plotly.react("ppg", ppgTraces, ppgLayout, {displayModeBar:false, responsive:true});

      const ibiLayout = darkLayoutBase("");
      ibiLayout.xaxis.title = "Time (s) (now = 0)";
      ibiLayout.yaxis.title = "IBI (ms)";
      ibiLayout.margin = {l:80,r:25,t:20,b:55};

      if (data.ibi_t && data.ibi_ms && data.ibi_t.length && data.ibi_ms.length) {
        const ibiTrace = {
          x: data.ibi_t,
          y: data.ibi_ms,
          mode: "lines+markers",
          name: "IBI",
          line: {width: 2.0},
          marker: {size: 6}
        };
        Plotly.react("ibi", [ibiTrace], ibiLayout, {displayModeBar:false, responsive:true});
      } else {
        Plotly.react("ibi", [], ibiLayout, {displayModeBar:false, responsive:true});
      }
    }

    if (hasEEG && hasBands && data.band_colors) {
      const bands = data.bands;
      const bandTraces = bands.map((b, bi) => ({
        type:"bar",
        name:b,
        x:data.channels,
        y:data.bp_frac.map(row => row[bi]),
        marker:{color: data.band_colors[b] || "#999"}
      }));
      const bandLayout = darkLayoutBase("Band power fraction (per channel)");
      bandLayout.barmode = "stack";
      bandLayout.margin = {l:70,r:25,t:45,b:45};
      bandLayout.yaxis = {range:[0,1], title:"Fraction", gridcolor:"__DARK_GRID__", zerolinecolor:"__DARK_GRID__"};
      Plotly.react("bands", bandTraces, bandLayout, {displayModeBar:false, responsive:true});
    }

    if (hasEEG && hasPSD && data.band_edges && data.bands && data.band_colors) {
      const psdTrace = { x:data.f, y:data.psd_db, mode:"lines", name:"Mean PSD", line:{width:2.2} };
      const shapes = [];
      const annotations = [];
      const ymax = Math.max(...data.psd_db);

      for (const b of data.bands) {
        const [f1, f2] = data.band_edges[b];
        const c = data.band_colors[b] || "#999";
        shapes.push({
          type:"rect", xref:"x", yref:"paper",
          x0:f1, x1:f2, y0:0, y1:1,
          fillcolor:c, opacity:0.14, line:{width:0}
        });
        annotations.push({
          x:(f1+f2)/2, y:ymax,
          xref:"x", yref:"y",
          text:b,
          showarrow:false,
          yanchor:"bottom",
          opacity:0.95,
          font: {size: 12, color: c}
        });
      }

      const psdLayout = darkLayoutBase("PSD (mean across channels)");
      psdLayout.xaxis.title = "Frequency (Hz)";
      psdLayout.yaxis.title = "PSD (dB)";
      psdLayout.margin = {l:80, r:25, t:45, b:70};
      psdLayout.shapes = shapes;
      psdLayout.annotations = annotations;
      Plotly.react("psd", [psdTrace], psdLayout, {displayModeBar:false, responsive:true});
    }
  } catch (e) {}
}

pollOnce();
window.__timer = setInterval(pollOnce, lastUpdateMs);
</script>
</body>
</html>
"""

def render_html(port: int) -> str:
    return (
        html_template
        .replace("__PORT__", str(int(port)))
        .replace("__DARK_BG__", DARK_BG)
        .replace("__DARK_PLOT__", DARK_PLOT)
        .replace("__DARK_GRID__", DARK_GRID)
        .replace("__DARK_TEXT__", DARK_TEXT)
    )