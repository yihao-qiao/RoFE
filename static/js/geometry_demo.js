(function () {
  "use strict";

  var root = document.getElementById("geometry-demo");
  var dataEl = document.getElementById("geometry-demo-data");
  var fallback = document.getElementById("geometry-static");
  if (!root || !dataEl || !document.createElementNS || !window.requestAnimationFrame) return;
  var D;
  try {
    D = JSON.parse(dataEl.textContent);
  } catch (err) {
    return;
  }

  var SVGNS = "http://www.w3.org/2000/svg";  // SVG namespace
  var INK = "#363636", GREY = "#9a9a9a", THEME = "#8C3A4B", PURPLE = "#6A4C93", RED = "#C33741";
  var DEG = Math.PI / 180;
  var ROT_END = 0.5;            // slider position where RoFE's rotation ends and the scaling starts
  var PLAY_MS = 4500;           // duration of one play-through
  var W = 360, H = 250;         // SVG viewBox of each panel
  var PC_ORDER = [1, 0, 2];     // data axes x, y, z are PC 2, PC 1, PC 3; bars and col labels follow PC 1, 2, 3
  var FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
  var EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ small linear algebra
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function unit(a) { var n = norm(a); return [a[0] / n, a[1] / n, a[2] / n]; }
  function rodrigues(axis, th) {  // rotation matrix about a unit axis
    var a = unit(axis), s = Math.sin(th), c = 1 - Math.cos(th);
    var K = [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]];
    var M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        var kk = 0;
        for (var k = 0; k < 3; k++) kk += K[i][k] * K[k][j];
        M[i][j] += s * K[i][j] + c * kk;
      }
    }
    return M;
  }
  function rowTimes(r, M) {  // row vector r times matrix M
    return [0, 1, 2].map(function (j) { return r[0] * M[0][j] + r[1] * M[1][j] + r[2] * M[2][j]; });
  }
  function std(values) {
    var m = values.reduce(function (s, v) { return s + v; }, 0) / values.length;
    return Math.sqrt(values.reduce(function (s, v) { return s + (v - m) * (v - m); }, 0) / values.length);
  }
  function smooth(x) { return x * x * (3 - 2 * x); }
  function clamp01(x) { return Math.max(0, Math.min(1, x)); }

  // The view of the paper's figure: fig:geometry's projection P turned by az about the vertical data axis, then
  // tilted by el about the horizontal axis across the line of sight (make_dispfield_fig2_data.view_projection).
  function view(az, el) {
    var P = D.view.P;
    var Rz = rodrigues([0, 0, 1], az * DEG);
    var pv = [rowTimes(P[0], Rz), rowTimes(P[1], Rz)];
    var t = unit(cross(pv[0], pv[1]));
    var Rt = rodrigues(cross([0, 0, 1], t), el * DEG);
    pv = [rowTimes(pv[0], Rt), rowTimes(pv[1], Rt)];
    return { x: pv[0], y: pv[1], toward: unit(cross(pv[0], pv[1])) };
  }

  // drawing scale and centre: fitted once, in the starting view, to everything any state draws (fitView below)
  var SCALE = 1, CX = W / 2, CY = H / 2;

  // ------------------------------------------------------------------ the states
  function rotationAt(u) {  // rows = RoFE's column axes after a fraction u of the rotation R
    var Ru = rodrigues(D.rotation.axis, u * D.rotation.angle);
    var rows = [0, 1, 2].map(function (k) { return [Ru[0][k], Ru[1][k], Ru[2][k]]; });
    return PC_ORDER.map(function (k) { return rows[k]; });
  }

  function state(method, t) {
    var u = clamp01(t / ROT_END), v = clamp01((t - ROT_END) / (1 - ROT_END));
    var f, axes, bars;
    if (method === "comet") {
      f = D.cometFactors.map(function (s) { return Math.pow(s, v); });
      axes = PC_ORDER.map(function (k) { var e = [0, 0, 0]; e[k] = 1; return e; });
    } else {
      var c = Math.pow(D.rofeFactor, v);
      f = [c, c, c];
      axes = rotationAt(u);
    }
    var pts = D.rows.map(function (r) { return [r[0] * f[0], r[1] * f[1], r[2] * f[2]]; });
    var q = [D.query[0] * f[0], D.query[1] * f[1], D.query[2] * f[2]];
    bars = axes.map(function (a) { return std(pts.map(function (p) { return dot(p, a); })); });
    return { pts: pts, q: q, axes: axes, bars: bars, u: u, v: v };
  }

  var BAR_MAX = Math.max.apply(null, state("comet", 0).bars);

  function fitView(vw) {
    var xs = [], ys = [], L = D.box;
    function add(p) { xs.push(dot(vw.x, p)); ys.push(dot(vw.y, p)); }
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      ["comet", "rofe"].forEach(function (m) {
        var s = state(m, t);
        s.pts.forEach(add);
        add(s.q);
      });
    });
    [0, 1, 2].forEach(function (k) {  // axis ends and their labels
      var e = [0, 0, 0]; e[k] = L * 1.1;
      add(e); add([-e[0], -e[1], -e[2]]);
    });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var pad = 16;
    SCALE = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
    CX = W / 2 - SCALE * (x0 + x1) / 2;
    CY = H / 2 + SCALE * (y0 + y1) / 2;
  }

  // ------------------------------------------------------------------ SVG helpers
  function el(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function label(parent, x, y, main, sub, attrs) {
    var t = el("text", Object.assign({ x: x.toFixed(1), y: y.toFixed(1) }, attrs || {}), parent);
    t.appendChild(document.createTextNode(main));
    if (sub) {
      var s = el("tspan", { dy: "3", "font-size": "0.72em" }, t);
      s.appendChild(document.createTextNode(sub));
    }
    return t;
  }
  // a five-pointed star with rounded tips and rounded inner corners: each corner is cut back by rc along both edges
  // and joined by a quadratic curve through the corner
  function starPath(x, y, ro, ri, rcOut, rcIn) {
    var v = [];
    for (var i = 0; i < 10; i++) {
      var r = i % 2 ? ri : ro, a = -Math.PI / 2 + i * Math.PI / 5;
      v.push([x + r * Math.cos(a), y + r * Math.sin(a)]);
    }
    function toward(p, q, d) {
      var dx = q[0] - p[0], dy = q[1] - p[1], len = Math.sqrt(dx * dx + dy * dy);
      return [p[0] + dx * d / len, p[1] + dy * d / len];
    }
    function f(p) { return p[0].toFixed(2) + "," + p[1].toFixed(2); }
    var d = "";
    for (var k = 0; k < 10; k++) {
      var p = v[k], prev = v[(k + 9) % 10], next = v[(k + 1) % 10], rc = k % 2 ? rcIn : rcOut;
      var a1 = toward(p, prev, rc), b1 = toward(p, next, rc);
      d += (k ? " L" : "M") + f(a1) + " Q" + f(p) + " " + f(b1);
    }
    return d + " Z";
  }

  // ------------------------------------------------------------------ one panel
  function Panel(node, method) {
    this.method = method;
    this.svg = node.querySelector("svg");
    this.barNodes = node.querySelectorAll(".gd-bar-fill");
    this.readout = node.querySelector(".gd-readout");
    this.svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  }

  Panel.prototype.draw = function (t, vw) {
    var svg = this.svg;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    function px(p) { return [CX + SCALE * dot(vw.x, p), CY - SCALE * dot(vw.y, p)]; }
    function depth(p) { return dot(vw.toward, p); }
    var s0 = state(this.method, 0), s = state(this.method, t);

    // principal axes (grey) and, for RoFE, the rotated column axes (purple)
    var L = D.box;
    var pcOpacity = this.method === "rofe" ? (1 - 0.65 * Math.min(1, s.u * 2)).toFixed(2) : "1";
    PC_ORDER.forEach(function (k, i) {
      var e = [0, 0, 0]; e[k] = L;
      var a = px([-e[0], -e[1], -e[2]]), b = px(e);
      el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: GREY, "stroke-width": 0.9, opacity: pcOpacity }, svg);
      var lb = px([e[0] * 1.1, e[1] * 1.1, e[2] * 1.1]);
      label(svg, lb[0], lb[1] + 4, "PC " + (i + 1), null,
            { class: "gd-axis-label", fill: GREY, opacity: pcOpacity, "text-anchor": "middle" });
    });
    if (this.method === "rofe" && s.u > 0.02) {
      var op = Math.min(1, s.u * 3).toFixed(2);
      s.axes.forEach(function (a, i) {
        var e = [a[0] * L * 0.8, a[1] * L * 0.8, a[2] * L * 0.8];
        var p0 = px([-e[0], -e[1], -e[2]]), p1 = px(e);
        el("line", { x1: p0[0], y1: p0[1], x2: p1[0], y2: p1[1], stroke: PURPLE, "stroke-width": 1.4,
                     opacity: op }, svg);
        var lb = px([e[0] * 1.13, e[1] * 1.13, e[2] * 1.13]);
        label(svg, lb[0], lb[1] + 4, "Col " + (i + 1), null,
              { class: "gd-axis-label gd-col-label", fill: PURPLE, opacity: op, "text-anchor": "middle" });
      });
    }

    // the shape before the scaling (dashed), then the current shape: faces far to near, then edges
    if (t > 0.001) {
      EDGES.forEach(function (ed) {
        var a = px(s0.pts[ed[0]]), b = px(s0.pts[ed[1]]);
        el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: GREY, "stroke-width": 1.3,
                     "stroke-dasharray": "4 3" }, svg);
      });
    }
    FACES.map(function (f) {
      return { f: f, d: (depth(s.pts[f[0]]) + depth(s.pts[f[1]]) + depth(s.pts[f[2]])) / 3 };
    }).sort(function (a, b) { return a.d - b.d; }).forEach(function (o) {
      var poly = o.f.map(function (i) { return px(s.pts[i]).map(function (v) { return v.toFixed(1); }).join(","); });
      el("polygon", { points: poly.join(" "), fill: THEME, "fill-opacity": 0.22, stroke: "none" }, svg);
    });
    EDGES.forEach(function (ed) {
      var a = px(s.pts[ed[0]]), b = px(s.pts[ed[1]]);
      el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: INK, "stroke-width": 2.3,
                   "stroke-linecap": "round" }, svg);
    });

    // nearest labeled example to the test row, linked by a red dashed line
    var dq = s.pts.map(function (p) { return norm([p[0] - s.q[0], p[1] - s.q[1], p[2] - s.q[2]]); });
    var near = dq.indexOf(Math.min.apply(null, dq));
    var qa = px(s.q), na = px(s.pts[near]);
    el("line", { x1: qa[0], y1: qa[1], x2: na[0], y2: na[1], stroke: RED, "stroke-width": 1.3,
                 "stroke-dasharray": "4 3" }, svg);

    // labeled examples (class 1: black circles, class 2: white squares), near to far drawn last
    var order = s.pts.map(function (p, i) { return i; }).sort(function (a, b) { return depth(s.pts[a]) - depth(s.pts[b]); });
    order.forEach(function (i) {
      var a = px(s.pts[i]);
      if (D.classes[i] === 1) {
        el("circle", { cx: a[0], cy: a[1], r: 5.2, fill: INK, stroke: "#fff", "stroke-width": 1.2 }, svg);
      } else {
        el("rect", { x: a[0] - 4.9, y: a[1] - 4.9, width: 9.8, height: 9.8, fill: "#fff", stroke: INK,
                     "stroke-width": 1.8 }, svg);
      }
      label(svg, a[0] + 7, a[1] - 7, "x", D.names[i].slice(1), { class: "gd-point-label", fill: INK });
    });

    // the test row: a red star in a white ring
    el("path", { d: starPath(qa[0], qa[1], 9, 4.6, 2.2, 1.2), fill: "#fff", stroke: "#fff", "stroke-width": 3.5,
                 "stroke-linejoin": "round" }, svg);
    el("path", { d: starPath(qa[0], qa[1], 9, 4.6, 2.2, 1.2), fill: RED, stroke: "#6e1f27", "stroke-width": 0.9,
                 "stroke-linejoin": "round" }, svg);

    // column standard deviations and the readout
    var self = this;
    s.bars.forEach(function (b, i) {
      if (self.barNodes[i]) self.barNodes[i].style.height = (100 * b / BAR_MAX).toFixed(1) + "%";
    });
    var dl = [];
    EDGES.forEach(function (ed) {
      var p = s.pts[ed[0]], r = s.pts[ed[1]];
      dl.push(norm([p[0] - r[0], p[1] - r[1], p[2] - r[2]]));
    });
    var lo = Math.min.apply(null, dl), hi = Math.max.apply(null, dl);
    var nearStart = D.nearest.scores;
    var nearText = "<i>x</i><sub>" + D.names[near].slice(1) + "</sub>" +
      (near !== nearStart ? " <span class=\"gd-changed\">(was <i>x</i><sub>" + D.names[nearStart].slice(1) +
       "</sub>)</span>" : "");
    var spread = hi / lo < 1.005 ? "all equal" : "longest " + (hi / lo).toFixed(2) + "&times; shortest";
    this.readout.innerHTML = "Distances: " + spread + "<br>Test row nearest: " + nearText;
  };

  // ------------------------------------------------------------------ wiring
  var panels = [new Panel(root.querySelector("[data-method=comet]"), "comet"),
                new Panel(root.querySelector("[data-method=rofe]"), "rofe")];
  var slider = root.querySelector(".gd-slider");
  var playButton = root.querySelector(".gd-play");
  var az0 = D.view.az, el0 = D.view.el, az = az0, elv = el0;
  var t = 0, playing = null;

  function render() {
    var vw = view(az, elv);
    panels.forEach(function (p) { p.draw(t, vw); });
    slider.value = Math.round(t * 1000);
    slider.setAttribute("aria-valuetext", t <= 0.001 ? "PCA scores" : t < ROT_END ? "RoFE rotates the columns" :
                        t >= 0.999 ? "after TabICL scales each column" : "TabICL scales each column");
  }

  function stop() {
    if (playing) cancelAnimationFrame(playing.raf);
    playing = null;
    playButton.textContent = "Play";
    playButton.setAttribute("aria-pressed", "false");
  }

  function play() {
    if (playing) { stop(); return; }
    if (t >= 0.999) t = 0;
    if (reduceMotion) { t = 1; render(); return; }
    var t0 = t, start = null;
    playButton.textContent = "Pause";
    playButton.setAttribute("aria-pressed", "true");
    playing = { raf: 0 };
    function step(now) {
      if (!playing) return;
      if (start === null) start = now;
      var x = clamp01((now - start) / (PLAY_MS * (1 - t0)));
      t = t0 + (1 - t0) * smooth(x);
      render();
      if (x < 1) playing.raf = requestAnimationFrame(step); else stop();
    }
    playing.raf = requestAnimationFrame(step);
  }

  slider.addEventListener("input", function () { stop(); t = Number(slider.value) / 1000; render(); });
  playButton.addEventListener("click", play);

  // drag either panel to turn the view of both; double-click resets it
  panels.forEach(function (p) {
    var drag = null;
    p.svg.addEventListener("pointerdown", function (e) {
      drag = { x: e.clientX, y: e.clientY, az: az, el: elv };
      p.svg.setPointerCapture(e.pointerId);
      p.svg.classList.add("is-dragging");
    });
    p.svg.addEventListener("pointermove", function (e) {
      if (!drag) return;
      az = drag.az + (e.clientX - drag.x) * 0.45;
      elv = Math.max(-15, Math.min(80, drag.el + (e.clientY - drag.y) * 0.35));
      render();
    });
    function end() { drag = null; p.svg.classList.remove("is-dragging"); }
    p.svg.addEventListener("pointerup", end);
    p.svg.addEventListener("pointercancel", end);
    p.svg.addEventListener("dblclick", function () { az = az0; elv = el0; render(); });
  });

  root.hidden = false;
  if (fallback) fallback.hidden = true;
  fitView(view(az0, el0));
  render();
})();
