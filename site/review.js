/* Court Watch review page: verdicts on flagged courts -> overrides.csv rows.
   Reads the same site/data files as the results page. Verdicts live in localStorage
   until copied out; nothing is sent anywhere. */
(async function () {
  const CLASSES = ["tennis", "hybrid", "pickleball", "padel", "removed"];
  const COLORS = { tennis: "#1f9e89", hybrid: "#bf8a18", pickleball: "#d2497f", padel: "#4f80d9", removed: "#a8623a", unknown: "#4b5563", unusable: "#4b5563" };
  const $ = id => document.getElementById(id);
  const parse = v => (typeof v === "string" ? JSON.parse(v) : v || []);
  const KEY = "courtwatch:review:v1";
  let verdicts = {};
  try { verdicts = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { verdicts = {}; }
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(verdicts)); } catch (e) {} ; progress(); };

  const [gj, sites] = await Promise.all([
    fetch("data/courts_points.geojson").then(r => r.json()).catch(() => ({ features: [] })),
    fetch("data/sites.json").then(r => r.json()).catch(() => ({})),
  ]);
  const courts = gj.features.map(f => ({ ...f.properties, lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1],
    history: parse(f.properties.history), transitions: parse(f.properties.transitions) }));

  const FILTERS = [
    ["flagged", "Flagged for review", c => c.needs_review],
    ["changed", "Any change", c => c.transitions.length > 0],
    ["hybrid", "Now hybrid", c => c.current_class === "hybrid"],
    ["pickleball", "Now pickleball", c => c.current_class === "pickleball"],
    ["removed", "Now removed", c => c.current_class === "removed"],
    ["all", "Everything", () => true],
  ];
  let filter = "flagged", shown = 0;
  const PAGE = 15;
  $("filters").innerHTML = FILTERS.map(([k, l, f]) => `<button class="chipbtn ${k === filter ? "on" : ""}" data-f="${k}">${l} <small>(${courts.filter(f).length})</small></button>`).join("");
  $("filters").onclick = e => { const b = e.target.closest("button"); if (!b) return; filter = b.dataset.f;
    document.querySelectorAll(".chipbtn").forEach(x => x.classList.toggle("on", x.dataset.f === filter)); shown = 0; $("list").innerHTML = ""; more(); };
  $("more").onclick = more;
  more(); progress();

  function list() {
    const f = FILTERS.find(x => x[0] === filter)[2];
    return courts.filter(f).sort((a, b) => (!!verdicts[a.court_id] - !!verdicts[b.court_id]) || (a.current_confidence - b.current_confidence));
  }
  function more() {
    const l = list(); const slice = l.slice(shown, shown + PAGE); shown += slice.length;
    if (!l.length) $("list").innerHTML = '<p class="empty">Nothing in this category.</p>';
    for (const c of slice) $("list").appendChild(card(c));
    $("more").hidden = shown >= l.length;
  }
  function progress() {
    const n = Object.keys(verdicts).length;
    $("progress").textContent = `${n} verdict${n === 1 ? "" : "s"} saved in this browser`;
  }

  function card(c) {
    const site = sites[c.site_id] || {};
    const geom = (site.courts || []).find(k => k.court_id === c.court_id);
    const [, trackId] = c.court_id.split(":");
    const el = document.createElement("div"); el.className = "rcard" + (verdicts[c.court_id] ? " done" : "");
    const changeYears = new Set(c.transitions.map(t => t.year_to));
    const reasons = [...new Set(c.transitions.filter(t => t.flagged).flatMap(t => String(t.flag_reasons).split(";")))].filter(Boolean);
    el.innerHTML = `
      <div class="rhead">
        <div><span class="id">${c.court_id}</span> · now <span class="tag ${c.current_class}">${c.current_class}</span> <small>conf ${Number(c.current_confidence).toFixed(2)}</small>${c.county && c.county !== "unknown" ? ` · ${c.county} County` : ""}</div>
        <div><span class="why">${reasons.length ? "⚑ " + reasons.join(", ").replace(/_/g, " ") : ""}</span>
          <a href="https://www.google.com/maps/@${c.lat},${c.lon},20z/data=!3m1!1e3" target="_blank" rel="noopener" style="color:#7fb4ff;margin-left:8px">live satellite ↗</a>
          <a href="map.html#${encodeURIComponent(c.court_id)}" style="color:#7fb4ff;margin-left:8px">map ↗</a></div>
      </div>
      <div class="strip">${c.history.map(h => `<figure class="${changeYears.has(h.year) ? "changed" : ""}" data-year="${h.year}"><canvas width="224" height="224"></canvas><figcaption><span>${h.year}${h.inferred ? " (no det.)" : ""}</span><span class="tag ${h.class}">${h.class}</span></figcaption></figure>`).join("")}</div>
      <div class="verdict">
        <span>Today this court is:</span>
        ${CLASSES.map(k => `<button data-cls="${k}">${k}</button>`).join("")}
        <button data-cls="unknown">can't tell</button>
        <button data-cls="__ok">model is right</button>
        <label>since <select class="since">${c.history.map(h => `<option value="${h.year}" ${h.year === c.history[c.history.length - 1].year ? "selected" : ""}>${h.year}</option>`).join("")}</select></label>
        <input class="note" placeholder="note (optional)" size="28">
      </div>`;
    const cvs = el.querySelectorAll("canvas");
    c.history.forEach((h, i) => drawZoom(cvs[i], site.images && site.images[h.year], geom && ((geom.history.find(x => x.year === h.year) || {}).obb || geom.obb), COLORS[h.class]));
    const v = verdicts[c.court_id];
    if (v) { el.querySelector(`button[data-cls="${v.cls}"]`)?.classList.add("on"); el.querySelector(".since").value = v.since; el.querySelector(".note").value = v.note || ""; }
    el.querySelector(".verdict").addEventListener("click", e => {
      const b = e.target.closest("button[data-cls]"); if (!b) return;
      el.querySelectorAll("button[data-cls]").forEach(x => x.classList.toggle("on", x === b));
      verdicts[c.court_id] = { cls: b.dataset.cls, since: +el.querySelector(".since").value, note: el.querySelector(".note").value,
        site_id: c.site_id, track_id: trackId, model_cls: c.current_class, years: c.history.map(h => h.year), model_by_year: Object.fromEntries(c.history.map(h => [h.year, h.class])) };
      el.classList.add("done"); save();
    });
    el.querySelector(".since").onchange = e => { if (verdicts[c.court_id]) { verdicts[c.court_id].since = +e.target.value; save(); } };
    el.querySelector(".note").oninput = e => { if (verdicts[c.court_id]) { verdicts[c.court_id].note = e.target.value; save(); } };
    return el;
  }

  function drawZoom(canvas, src, obb, color) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!src) { ctx.fillStyle = "#666"; ctx.font = "12px system-ui"; ctx.fillText("no image", 80, 116); return; }
    const img = new Image();
    img.onload = () => {
      const W = img.naturalWidth, H = img.naturalHeight;
      let cx = W / 2, cy = H / 2, size = Math.min(W, H) / 2;
      if (obb) {
        const xs = obb.map(p => p[0] * W), ys = obb.map(p => p[1] * H);
        cx = (Math.min(...xs) + Math.max(...xs)) / 2; cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        size = Math.max(90, Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.8);
      }
      const sx = Math.max(0, Math.min(W - size, cx - size / 2)), sy = Math.max(0, Math.min(H - size, cy - size / 2));
      ctx.drawImage(img, sx, sy, size, size, 0, 0, canvas.width, canvas.height);
      if (obb) { const k = canvas.width / size; ctx.beginPath(); obb.forEach(([x, y], i) => { const px = (x * W - sx) * k, py = (y * H - sy) * k; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.closePath(); ctx.lineWidth = 2; ctx.strokeStyle = color || "#fff"; ctx.stroke(); }
    };
    img.src = src;
  }

  // ---- export ----------------------------------------------------------------
  function csv() {
    const esc = s => `"${String(s ?? "").replace(/"/g, '""')}"`;
    const lines = ["site_id,track_id,year,class,reviewer,note"];
    for (const v of Object.values(verdicts)) {
      const years = v.years.filter(y => y >= v.since);
      for (const y of years) {
        const cls = v.cls === "__ok" ? v.model_by_year[y] : v.cls;
        const note = v.cls === "__ok" ? ("confirmed" + (v.note ? "; " + v.note : "")) : (v.note || "");
        lines.push([v.site_id, v.track_id, y, cls, "scott", esc(note)].join(","));
      }
    }
    return lines.join("\n") + "\n";
  }
  $("show").onclick = () => { const t = $("out"); t.value = csv(); t.hidden = !t.hidden; };
  $("copy").onclick = async () => {
    const text = csv();
    try { await navigator.clipboard.writeText(text); $("copy").textContent = "Copied!"; setTimeout(() => ($("copy").textContent = "Copy corrections"), 1500); }
    catch (e) { $("out").value = text; $("out").hidden = false; }
  };
  $("clear").onclick = () => { if (confirm("Clear all saved verdicts?")) { verdicts = {}; save(); shown = 0; $("list").innerHTML = ""; more(); } };
})();
