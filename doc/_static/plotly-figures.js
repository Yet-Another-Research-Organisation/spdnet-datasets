/* Interactive dataset figures (doc/_ext/dataset_figures.py).
 *
 * Each <div class="spd-plotly-div" data-src="....json"> holds a Plotly figure
 * written by doc/tools/figures_io.py. When the first one scrolls into view,
 * plotly.js (cartesian bundle, pinned version, subresource integrity) is loaded
 * once for the page; each figure's JSON is fetched when it becomes visible.
 * Colours follow the light / dark theme of pydata-sphinx-theme, also when the
 * reader switches theme.
 */
(function () {
  "use strict";
  var PLOTLY_URL = "https://cdn.plot.ly/plotly-cartesian-4.1.1.min.js";
  var PLOTLY_SRI = "sha384-zc4SwKObGL/W0M3RUI09UPWjlgIZHfbwMgcZO6713mRh78tfUmIQKhzCyE2LIiUg";

  // dataviz reference palette: light steps -> dark steps
  var DARK_STEP = {
    "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70",
    "#eda100": "#c98500", "#e87ba4": "#d55181", "#008300": "#0ca30c",
    "#6250d6": "#9085e9", "#e34948": "#e66767"
  };
  var THEME = {
    light: { ink: "#0b0b0b", muted: "#52514e", grid: "#e7e6e2", mid: "#f0efec", fill: "rgba(42,120,214,0.18)" },
    dark: { ink: "#f0efec", muted: "#c3c2b7", grid: "#34342f", mid: "#383835", fill: "rgba(57,135,229,0.25)" }
  };

  var plotlyPromise = null;
  function loadPlotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (!plotlyPromise) {
      plotlyPromise = new Promise(function (resolve, reject) {
        var s = document.createElement("script");
        s.src = PLOTLY_URL;
        s.integrity = PLOTLY_SRI;
        s.crossOrigin = "anonymous";
        s.onload = function () { resolve(window.Plotly); };
        s.onerror = reject;
        document.head.appendChild(s);
      });
    }
    return plotlyPromise;
  }

  function mode() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function themed(fig, m) {
    var t = THEME[m];
    var data = fig.data.map(function (tr) {
      var c = JSON.parse(JSON.stringify(tr));
      var swap = function (col) {
        return (m === "dark" && typeof col === "string" && DARK_STEP[col.toLowerCase()]) || col;
      };
      if (c.line && c.line.color) c.line.color = swap(c.line.color);
      if (c.marker && c.marker.color) c.marker.color = swap(c.marker.color);
      if (c.fill === "toself") c.fillcolor = t.fill;
      if (Array.isArray(c.colorscale)) {
        c.colorscale = c.colorscale.map(function (s) {
          return [s[0], String(s[1]).toLowerCase() === "#f0efec" ? t.mid : s[1]];
        });
        c.colorbar = Object.assign({}, c.colorbar, { tickfont: { color: t.muted }, outlinewidth: 0 });
      }
      return c;
    });
    var layout = JSON.parse(JSON.stringify(fig.layout || {}));
    layout.paper_bgcolor = "rgba(0,0,0,0)";
    layout.plot_bgcolor = "rgba(0,0,0,0)";
    layout.font = Object.assign({}, layout.font, { color: t.muted });
    if (layout.title) {
      layout.title = Object.assign({}, layout.title, {
        x: 0.02, xanchor: "left",
        font: Object.assign({}, layout.title.font, { color: t.ink })
      });
    }
    (layout.annotations || []).forEach(function (a) {
      a.font = Object.assign({}, a.font, { color: t.ink, size: 13 });
    });
    Object.keys(layout).forEach(function (k) {
      if (/^[xy]axis\d*$/.test(k)) {
        layout[k] = Object.assign({}, layout[k], { gridcolor: t.grid, zeroline: false, linecolor: t.grid });
      }
    });
    if (!Object.keys(layout).some(function (k) { return /^[xy]axis/.test(k); })) {
      layout.xaxis = { gridcolor: t.grid };
      layout.yaxis = { gridcolor: t.grid };
    }
    layout.autosize = true;
    layout.hoverlabel = { bgcolor: m === "dark" ? "#1b2028" : "#ffffff", font: { color: t.ink } };
    return { data: data, layout: layout };
  }

  var rendered = [];
  function render(div) {
    Promise.all([loadPlotly(), fetch(div.dataset.src).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + div.dataset.src);
      return r.json();
    })]).then(function (res) {
      var Plotly = res[0], fig = res[1];
      div._spdFigure = fig;
      var f = themed(fig, mode());
      return Plotly.newPlot(div, f.data, f.layout, {
        responsive: true, displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"]
      }).then(function () { rendered.push(div); });
    }).catch(function (err) {
      div.textContent = "This interactive figure could not be loaded (" + err.message + ").";
    });
  }

  function retheme() {
    if (!window.Plotly) return;
    var m = mode();
    rendered.forEach(function (div) {
      var f = themed(div._spdFigure, m);
      window.Plotly.react(div, f.data, f.layout);
    });
  }

  function init() {
    var divs = document.querySelectorAll(".spd-plotly-div[data-src]");
    if (!divs.length) return;
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { io.unobserve(e.target); render(e.target); }
        });
      }, { rootMargin: "300px 0px" });
      divs.forEach(function (d) { io.observe(d); });
    } else {
      divs.forEach(render);
    }
    new MutationObserver(retheme).observe(document.documentElement,
      { attributes: true, attributeFilter: ["data-theme"] });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
