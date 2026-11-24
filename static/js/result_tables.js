(function () {
  "use strict";

  function makeGroup(label, options, onPick) {
    var wrap = document.createElement("div");
    wrap.className = "rt-group";
    var name = document.createElement("span");
    name.className = "rt-group-label";
    name.textContent = label;
    wrap.appendChild(name);
    var bar = document.createElement("div");
    bar.className = "buttons has-addons";
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", label);
    var buttons = options.map(function (opt, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "button is-small is-rounded";
      b.textContent = opt;
      b.addEventListener("click", function () { onPick(i); });
      bar.appendChild(b);
      return b;
    });
    wrap.appendChild(bar);
    return { node: wrap, select: function (k) {
      buttons.forEach(function (b, i) {
        b.classList.toggle("is-dark", i === k);
        b.setAttribute("aria-pressed", i === k ? "true" : "false");
      });
    } };
  }

  Array.prototype.forEach.call(document.querySelectorAll("table.result-table"), function (table) {
    var groups = table.querySelectorAll("tr.group[data-block]");
    var blocks = Array.prototype.map.call(groups, function (row) {
      var title = row.textContent.trim(), cut = title.lastIndexOf(", ");
      return { id: row.getAttribute("data-block"), title: title,
               input: cut > 0 ? title.slice(0, cut) : title, encoder: cut > 0 ? title.slice(cut + 2) : "" };
    });
    if (blocks.length < 2) return;
    var inputs = [], encoders = [];
    blocks.forEach(function (b) {
      if (inputs.indexOf(b.input) < 0) inputs.push(b.input);
      if (b.encoder && encoders.indexOf(b.encoder) < 0) encoders.push(b.encoder);
    });
    var pick = { input: 0, encoder: 0 };
    var rows = table.querySelectorAll("tbody tr[data-block]");

    function show() {
      var want = blocks.filter(function (b) {
        return b.input === inputs[pick.input] && (!encoders.length || b.encoder === encoders[pick.encoder]);
      })[0];
      Array.prototype.forEach.call(rows, function (r) {
        var inBlock = want && r.getAttribute("data-block") === want.id;
        r.hidden = !inBlock || r.classList.contains("group");  // the buttons name the block, so its title row hides
      });
      inputGroup.select(pick.input);
      if (encoderGroup) encoderGroup.select(pick.encoder);
    }

    var controls = document.createElement("div");
    controls.className = "rt-controls";
    var caption = table.querySelector("caption");
    if (caption) {  // the table's title moves above the buttons
      var title = document.createElement("p");
      title.className = "rt-title";
      title.textContent = caption.textContent;
      controls.appendChild(title);
      caption.hidden = true;
    }
    var inputGroup = makeGroup("Input", inputs, function (i) { pick.input = i; show(); });
    controls.appendChild(inputGroup.node);
    var encoderGroup = null;
    if (encoders.length > 1) {
      encoderGroup = makeGroup("Encoder", encoders, function (i) { pick.encoder = i; show(); });
      controls.appendChild(encoderGroup.node);
    }
    var container = table.closest(".table-container") || table;
    container.parentNode.insertBefore(controls, container);
    table.classList.add("is-one-block");
    show();
  });
})();
