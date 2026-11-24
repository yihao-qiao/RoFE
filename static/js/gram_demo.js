(function () {
  "use strict";

  var root = document.getElementById("gram-demo");
  var dataEl = document.getElementById("gram-demo-data");
  if (!root || !dataEl || !window.requestAnimationFrame) return;
  var D;
  try {
    D = JSON.parse(dataEl.textContent);
  } catch (err) {
    return;
  }

  var SVGNS = "http://www.w3.org/2000/svg";  // SVG namespace
  var THEME = [140, 58, 75], WHITE = [255, 255, 255], INK = "#363636", GREY = "#9a9a9a";
  var VMIN = -0.4, VMAX = 0.8, KNEE = 0.35;  // the paper figure's colour map: white, themec!18 at 0.35, themec
  var PLAY_MS = 4500;
  var n = D.n, w = D.w, F = D.frames.length;
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ values at a slider position
  function mix(a, b, t) { return a + (b - a) * t; }
  function at(u) {
    var x = u * (F - 1), k = Math.min(F - 2, Math.floor(x)), t = x - k;
    var A = D.frames[k], B = D.frames[k + 1];
    return {
      cos: A.cos.map(function (v, i) { return mix(v, B.cos[i], t) / 100; }),
      v: A.var.map(function (v, i) { return mix(v, B.var[i], t); }),
      cv: mix(A.cv, B.cv, t),
      agree: A.agree === null || B.agree === null ? (t < 0.5 ? A.agree : B.agree) : mix(A.agree, B.agree, t),
      first: k === 0 && t < 1e-6
    };
  }

  // ------------------------------------------------------------------ heat maps
  function colour(v) {
    var p = Math.max(0, Math.min(1, (v - VMIN) / (VMAX - VMIN)));
    var light = WHITE.map(function (c, i) { return c + (THEME[i] - c) * 0.18; });
    var from, to, s;
    if (p <= KNEE) { from = WHITE; to = light; s = p / KNEE; } else { from = light; to = THEME; s = (p - KNEE) / (1 - KNEE); }
    return "rgb(" + from.map(function (c, i) { return Math.round(c + (to[i] - c) * s); }).join(",") + ")";
  }
  function paint(canvas, upper) {
    var ctx = canvas.getContext("2d");
    var idx = 0;
    for (var i = 0; i < n; i++) {
      ctx.fillStyle = colour(1);
      ctx.fillRect(i, i, 1, 1);
      for (var j = i + 1; j < n; j++) {
        ctx.fillStyle = colour(upper[idx++]);
        ctx.fillRect(j, i, 1, 1);
        ctx.fillRect(i, j, 1, 1);
      }
    }
  }

  // ------------------------------------------------------------------ column variances
  var VAR_MAX = Math.max.apply(null, D.frames.map(function (f) { return Math.max.apply(null, f.var); }));
  // the chart is laid out in screen pixels (layout() below): its x axis sits on the bottom edge of the heat maps
  var BW = 300, BH = 150, PADL = 30, PADR = 34, PADB = 22, PADT = 8;
  function el(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, s, attrs) {
    var t = el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.appendChild(document.createTextNode(s));
    return t;
  }
  function drawBars(svg, v) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var plotW = BW - PADL - PADR, plotH = BH - PADB - PADT, step = plotW / w;
    function yOf(val) { return PADT + plotH * (1 - val / VAR_MAX); }
    [0, 10, 20, 30].forEach(function (tick) {
      if (tick > VAR_MAX) return;
      var yy = yOf(tick);
      el("line", { x1: PADL, x2: PADL + plotW, y1: yy, y2: yy, stroke: "#e6e6e6", "stroke-width": 0.8 }, svg);
      text(svg, PADL - 5, yy + 3.5, String(tick), { class: "gm-tick", "text-anchor": "end", fill: GREY });
    });
    v.forEach(function (val, k) {
      var h = plotH * val / VAR_MAX;
      el("rect", { x: (PADL + k * step + step * 0.15).toFixed(2), y: (PADT + plotH - h).toFixed(2),
                   width: (step * 0.7).toFixed(2), height: Math.max(0, h).toFixed(2), fill: GREY }, svg);
    });
    var ym = yOf(D.meanVar);
    el("line", { x1: PADL, x2: PADL + plotW, y1: ym, y2: ym, stroke: INK, "stroke-width": 1.1, "stroke-dasharray": "4 3" }, svg);
    text(svg, PADL + plotW + 4, ym + 3, "mean", { class: "gm-tick", "text-anchor": "start", fill: INK });
    el("line", { x1: PADL, x2: PADL + plotW, y1: PADT + plotH - 0.5, y2: PADT + plotH - 0.5, stroke: INK,
                 "stroke-width": 1 }, svg);
    text(svg, PADL + step / 2, BH - 6, "1", { class: "gm-tick", "text-anchor": "middle", fill: GREY });
    text(svg, PADL + plotW - step / 2, BH - 6, String(w), { class: "gm-tick", "text-anchor": "middle", fill: GREY });
    text(svg, PADL + plotW / 2, BH - 6, "column", { class: "gm-tick", "text-anchor": "middle", fill: GREY });
  }

  // ------------------------------------------------------------------ wiring
  var pcaCanvas = root.querySelector("canvas[data-which=pca]");
  var curCanvas = root.querySelector("canvas[data-which=current]");
  var barSvg = root.querySelector("svg.gm-bars");
  var cosOut = root.querySelector(".gm-readout-cos");
  var varOut = root.querySelector(".gm-readout-var");
  var slider = root.querySelector(".gm-slider");
  var playButton = root.querySelector(".gm-play");
  [pcaCanvas, curCanvas].forEach(function (c) { c.width = n; c.height = n; });
  paint(pcaCanvas, D.pca.map(function (v) { return v / 100; }));

  // size the chart so that one unit is one screen pixel and the x axis lies on the heat maps' bottom edge (both panels
  // start under a one-line title); when the panels wrap onto separate rows the chart keeps the heat maps' height
  function layout() {
    var heat = curCanvas.getBoundingClientRect(), width = barSvg.getBoundingClientRect().width;
    if (!heat.height || !width) return false;
    BW = Math.round(width);
    BH = Math.round(heat.height) + PADB;
    PADT = 8;
    barSvg.setAttribute("viewBox", "0 0 " + BW + " " + BH);
    barSvg.style.height = BH + "px";
    return true;
  }

  var u = 0, playing = null;
  function render() {
    var s = at(u);
    paint(curCanvas, s.cos);
    drawBars(barSvg, s.v);
    cosOut.innerHTML = s.first || s.agree === null
      ? "All similarities " + D.simplex.toFixed(2).replace("-", "&minus;") + ": a regular simplex"
      : "<span class=\"nowrap\"><i>r</i> = " + s.agree.toFixed(2) + "</span> with the PCA scores";
    varOut.innerHTML = "Coefficient of variation: " + s.cv.toFixed(2);
    slider.value = Math.round(u * 1000);
    slider.setAttribute("aria-valuetext", u <= 0.001 ? "principal axes (CoMET)" :
                        u >= 0.999 ? "rotated by R (RoFE)" : "rotation " + Math.round(u * 100) + "% of the way");
  }

  function stop() {
    if (playing) cancelAnimationFrame(playing.raf);
    playing = null;
    playButton.textContent = "Play";
    playButton.setAttribute("aria-pressed", "false");
  }
  function play() {
    if (playing) { stop(); return; }
    if (u >= 0.999) u = 0;
    if (reduceMotion) { u = 1; render(); return; }
    var u0 = u, start = null;
    playButton.textContent = "Pause";
    playButton.setAttribute("aria-pressed", "true");
    playing = { raf: 0 };
    function step(now) {
      if (!playing) return;
      if (start === null) start = now;
      var x = Math.max(0, Math.min(1, (now - start) / (PLAY_MS * (1 - u0))));
      u = u0 + (1 - u0) * x * x * (3 - 2 * x);
      render();
      if (x < 1) playing.raf = requestAnimationFrame(step); else stop();
    }
    playing.raf = requestAnimationFrame(step);
  }
  slider.addEventListener("input", function () { stop(); u = Number(slider.value) / 1000; render(); });
  playButton.addEventListener("click", play);

  root.hidden = false;
  var note = document.getElementById("gram-demo-noscript");
  if (note) note.hidden = true;
  layout();
  render();
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () { if (layout()) render(); }, 80);
  });
  window.addEventListener("load", function () { if (layout()) render(); });
})();
