/* Court Watch results page. Reads site/data/ written by scripts/07_export.py:
   summary.json, courts_points.geojson, sites.json, chips/<site>/<year>.jpg. No backend. */
(async function () {
  const CLASSES = ["tennis", "hybrid", "pickleball", "padel", "removed", "unknown"];
  const COLORS = { tennis: "#1f9e89", hybrid: "#bf8a18", pickleball: "#d2497f", padel: "#4f80d9", removed: "#a8623a", unknown: "#4b5563" };
  const LABEL = { tennis: "still tennis", hybrid: "hybrid: tennis with pickleball lines", pickleball: "converted to pickleball", padel: "padel", removed: "removed", unknown: "unknown" };
  const $ = id => document.getElementById(id);
  const fmt = v => (v === undefined || v === null || Number.isNaN(Number(v)) ? "–" : Number(v).toLocaleString());
  const parse = v => (typeof v === "string" ? JSON.parse(v) : v || []);

  const [summary, gj, sites] = await Promise.all([
    fetch("data/summary.json").then(r => r.json()).catch(() => ({})),
    fetch("data/courts_points.geojson").then(r => r.json()).catch(() => ({ features: [] })),
    fetch("data/sites.json").then(r => r.json()).catch(() => ({})),
  ]);
  const courts = gj.features.map(f => ({ ...f.properties, lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1],
    history: parse(f.properties.history), transitions: parse(f.properties.transitions) }));

  if (summary.demo) {
    const b = document.createElement("div"); b.id = "demo-banner";
    b.textContent = "Demo with synthetic data. These are not real courts or real numbers; the Ohio run has not been published yet.";
    document.body.prepend(b);
  }
  const stateName = { OH: "Ohio" }[summary.state] || summary.state || "Ohio";
  document.querySelectorAll(".state").forEach(e => (e.textContent = stateName));

  // ---- hero + tiles ----------------------------------------------------------
  const counts = Object.fromEntries(CLASSES.map(c => [c, courts.filter(x => x.current_class === c).length]));
  const pbCourts = courts.filter(x => x.current_class === "pickleball").reduce((a, x) => a + (x.current_n_courts ?? 0), 0);
  const pbUnknown = courts.filter(x => x.current_class === "pickleball" && (x.current_n_courts === null || x.current_n_courts === undefined)).length;
  const years = [...new Set(courts.flatMap(c => c.history.map(h => h.year)))].sort();
  $("hero-num").textContent = fmt(counts.hybrid);
  $("hero-fine").textContent = courts.length
    ? `${fmt(courts.length)} court footprints at ${fmt(Object.keys(sites).length)} sites, imagery ${years[0]} to ${years[years.length - 1]}.`
    : "No results published yet.";
  const flagged = courts.filter(c => c.needs_review).length;
  const tiles = [
    ["tennis", counts.tennis, LABEL.tennis],
    ["pickleball", counts.pickleball, `converted footprints${pbCourts ? ` (${fmt(pbCourts)} pickleball courts${pbUnknown ? `, ${pbUnknown} uncounted` : ""})` : ""}`],
    ["removed", counts.removed, "removed"],
    ["padel", counts.padel, "padel"],
    ["unknown", counts.unknown, "unknown this year"],
    [null, flagged, "flagged for human review"],
  ];
  $("tiles").innerHTML = tiles.map(([c, v, k]) =>
    `<div class="tile"><div class="v">${fmt(v)}</div><div class="k">${c ? `<span class="dot" style="background:${COLORS[c]}"></span>` : "⚑ "}${k}</div></div>`).join("");
  $("generated").textContent = summary.generated_at ? `Updated ${summary.generated_at.slice(0, 10)}.` : "";

  // ---- chart: transitions by year_to, stacked by kind -------------------------
  const KINDS = [
    { key: "to_hybrid", label: "tennis → hybrid", color: COLORS.hybrid, test: t => t.from === "tennis" && t.to === "hybrid" },
    { key: "to_pickleball", label: "→ pickleball", color: COLORS.pickleball, test: t => t.to === "pickleball" && t.from !== "pickleball" },
    { key: "to_padel", label: "→ padel", color: COLORS.padel, test: t => t.to === "padel" && t.from !== "padel" },
    { key: "to_removed", label: "→ removed", color: COLORS.removed, test: t => t.to === "removed" },
    { key: "back_to_tennis", label: "back to tennis (flagged)", color: COLORS.tennis, test: t => t.to === "tennis" },
  ];
  const byYear = {};
  for (const c of courts) for (const t of c.transitions) {
    const k = KINDS.find(k => k.test(t)); if (!k) continue;
    (byYear[t.year_to] ||= Object.fromEntries(KINDS.map(k => [k.key, 0])))[k.key]++;
  }
  const chartYears = Object.keys(byYear).map(Number).sort();
  $("chart-legend").innerHTML = KINDS.map(k => `<span><span class="dot" style="background:${k.color}"></span>${k.label}</span>`).join("");
  drawChart(chartYears, byYear);
  $("chart-table").innerHTML = chartYears.length ? `<table class="tbl"><thead><tr><th>Year</th>${KINDS.map(k => `<th>${k.label}</th>`).join("")}<th>Total</th></tr></thead><tbody>${
    chartYears.map(y => `<tr><td>${y}</td>${KINDS.map(k => `<td>${byYear[y][k.key]}</td>`).join("")}<td>${Object.values(byYear[y]).reduce((a, b) => a + b, 0)}</td></tr>`).join("")}</tbody></table>` : "";
  $("chart-toggle").onclick = () => {
    const t = $("chart-table"); t.hidden = !t.hidden; $("chart").hidden = !t.hidden;
    $("chart-toggle").textContent = t.hidden ? "Show as table" : "Show as chart";
  };

  function drawChart(yearsX, data) {
    const el = $("chart");
    if (!yearsX.length) { el.innerHTML = '<p class="empty">No class changes detected yet.</p>'; return; }
    const W = 900, H = 260, m = { l: 36, r: 10, t: 10, b: 28 };
    const totals = yearsX.map(y => Object.values(data[y]).reduce((a, b) => a + b, 0));
    const maxV = Math.max(1, ...totals);
    const step = niceStep(maxV);
    const top = Math.ceil(maxV / step) * step;
    const slot = (W - m.l - m.r) / yearsX.length;
    const x = i => m.l + (i + 0.5) * slot;
    const yScale = v => m.t + (H - m.t - m.b) * (1 - v / top);
    const barW = Math.min(24, 0.6 * slot);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Court class changes by year">`;
    svg += '<g class="grid">';
    for (let v = 0; v <= top; v += step) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${yScale(v)}" y2="${yScale(v)}"/><text x="${m.l - 6}" y="${yScale(v) + 4}" text-anchor="end">${v.toLocaleString()}</text>`;
    svg += "</g>";
    yearsX.forEach((y, i) => {
      let acc = 0;
      const segs = KINDS.filter(k => data[y][k.key] > 0);
      segs.forEach((k, si) => {
        const v = data[y][k.key]; const y0 = yScale(acc), y1 = yScale(acc + v); acc += v;
        const last = si === segs.length - 1;
        const h = Math.max(0, y0 - y1 - (last ? 0 : 2));   // 2px surface gap between stacked segments
        const r = Math.min(4, h / 2, barW / 2);
        const x0 = x(i) - barW / 2;
        const path = last
          ? `M${x0},${y0} v${-(h - r)} a${r},${r} 0 0 1 ${r},${-r} h${barW - 2 * r} a${r},${r} 0 0 1 ${r},${r} v${h - r} z`
          : `M${x0},${y0} v${-h} h${barW} v${h} z`;
        svg += `<path d="${path}" fill="${k.color}"/>`;
      });
      svg += `<text class="cap" x="${x(i)}" y="${yScale(totals[i]) - 6}" text-anchor="middle">${totals[i]}</text>`;
      svg += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${y}</text>`;
      svg += `<rect x="${x(i) - slot / 2}" y="${m.t}" width="${slot}" height="${H - m.t - m.b}" fill="transparent" data-col="${y}"/>`;
    });
    svg += "</svg>";
    el.innerHTML = svg;
    const tip = $("tip");
    el.querySelectorAll("rect[data-col]").forEach(r => {
      r.addEventListener("mousemove", e => {
        const y = r.dataset.col;
        tip.innerHTML = `<b>${y}</b>` + KINDS.filter(k => data[y][k.key]).map(k => `<div><span class="dot" style="background:${k.color}"></span> ${k.label}: ${data[y][k.key]}</div>`).join("");
        tip.style.display = "block"; tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 14) + "px";
      });
      r.addEventListener("mouseleave", () => (tip.style.display = "none"));
    });
  }
  function niceStep(maxV) { const raw = maxV / 4; const p = Math.pow(10, Math.floor(Math.log10(raw))); const n = raw / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; }

  // ---- gallery ----------------------------------------------------------------
  const FILTERS = [
    ["changed", "All changes", c => c.transitions.length > 0],
    ["hybrid", "Became hybrid", c => c.transitions.some(t => t.to === "hybrid")],
    ["pickleball", "Became pickleball", c => c.transitions.some(t => t.to === "pickleball" && t.from !== "pickleball")],
    ["removed", "Removed", c => c.transitions.some(t => t.to === "removed")],
    ["review", "Flagged for review", c => c.needs_review],
  ];
  let filter = "changed", shown = 0;
  const PAGE = 12;
  $("gallery-filters").innerHTML = FILTERS.map(([k, l, f]) => `<button class="chipbtn ${k === filter ? "on" : ""}" data-f="${k}">${l} <small>(${courts.filter(f).length})</small></button>`).join("");
  $("gallery-filters").onclick = e => { const b = e.target.closest("button"); if (!b) return; filter = b.dataset.f;
    document.querySelectorAll(".chipbtn").forEach(x => x.classList.toggle("on", x.dataset.f === filter)); shown = 0; $("cards").innerHTML = ""; renderMore(); };
  $("more").onclick = renderMore;
  renderMore();

  function candidates() {
    const f = FILTERS.find(x => x[0] === filter)[2];
    return courts.filter(f).sort((a, b) => (a.needs_review - b.needs_review) || (b.current_confidence - a.current_confidence));
  }
  function renderMore() {
    const list = candidates();
    if (!list.length && shown === 0) { $("cards").innerHTML = '<p class="empty">Nothing in this category yet.</p>'; $("more").hidden = true; return; }
    const slice = list.slice(shown, shown + PAGE); shown += slice.length;
    for (const c of slice) $("cards").appendChild(card(c));
    $("more").hidden = shown >= list.length;
  }
  function card(c) {
    const site = sites[c.site_id] || {};
    const first = c.history.find(h => !h.inferred) || c.history[0];
    const last = c.history[c.history.length - 1];   // latest imagery, even if the class there was inferred (removed)
    const a = document.createElement("a"); a.className = "gcard"; a.href = `map.html#${encodeURIComponent(c.court_id)}`;
    const t = c.transitions[c.transitions.length - 1];
    a.innerHTML = `<div class="pair">${[first, last].map(h => `<figure><canvas width="240" height="240"></canvas><figcaption>${h.year} · ${h.class}${h.class === "pickleball" && h.n_courts > 1 ? " ×" + h.n_courts : ""}${h.inferred ? " (inferred)" : ""}</figcaption></figure>`).join("")}</div>
      <div class="meta"><div class="trans">${t ? `<span class="tag ${t.from}">${t.from}</span> → <span class="tag ${t.to}">${t.to}</span> <small>between ${t.year_from} and ${t.year_to}</small>` : `<span class="tag ${c.current_class}">${c.current_class}</span> unchanged`}</div>
      ${c.county && c.county !== "unknown" ? c.county + " County · " : ""}${c.history.length} imagery years · confidence ${Number(c.current_confidence).toFixed(2)}${c.needs_review ? ' · <span class="flag">⚑ flagged</span>' : ""}${c.n_overrides ? " · reviewed" : ""}</div>`;
    const cvs = a.querySelectorAll("canvas");
    const courtGeom = (site.courts || []).find(k => k.court_id === c.court_id);
    [first, last].forEach((h, i) => drawZoom(cvs[i], site.images && site.images[h.year], (courtGeom && ((courtGeom.history.find(x => x.year === h.year) || {}).obb || courtGeom.obb)), COLORS[h.class]));
    return a;
  }
  function drawZoom(canvas, src, obb, color) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!src) { ctx.fillStyle = "#666"; ctx.font = "12px system-ui"; ctx.fillText("no image", 90, 124); return; }
    const img = new Image();
    img.onload = () => {
      const W = img.naturalWidth, H = img.naturalHeight;
      let cx = W / 2, cy = H / 2, size = Math.min(W, H) / 2;
      if (obb) {
        const xs = obb.map(p => p[0] * W), ys = obb.map(p => p[1] * H);
        cx = (Math.min(...xs) + Math.max(...xs)) / 2; cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 2.2;
        size = Math.max(size, 90);
      }
      const sx = Math.max(0, Math.min(W - size, cx - size / 2)), sy = Math.max(0, Math.min(H - size, cy - size / 2));
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, sx, sy, size, size, 0, 0, canvas.width, canvas.height);
      if (obb) {
        const k = canvas.width / size;
        ctx.beginPath(); obb.forEach(([x, y], i) => { const px = (x * W - sx) * k, py = (y * H - sy) * k; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.closePath();
        ctx.lineWidth = 2.5; ctx.strokeStyle = color || "#fff"; ctx.stroke();
      }
    };
    img.src = src;
  }

  // ---- counties ---------------------------------------------------------------
  const byCounty = {};
  for (const c of courts) {
    const k = c.county && c.county !== "unknown" ? c.county : "Unknown county";
    const row = (byCounty[k] ||= { total: 0, tennis: 0, hybrid: 0, pickleball: 0, padel: 0, removed: 0, unknown: 0, flagged: 0 });
    row.total++; row[c.current_class in row ? c.current_class : "unknown"]++; if (c.needs_review) row.flagged++;
  }
  const rows = Object.entries(byCounty).sort((a, b) => (b[1].hybrid + b[1].pickleball) - (a[1].hybrid + a[1].pickleball) || b[1].total - a[1].total);
  $("county-table").innerHTML = rows.length ? `<thead><tr><th>County</th><th>Footprints</th>${["tennis", "hybrid", "pickleball", "padel", "removed"].map(c => `<th><span class="dot" style="background:${COLORS[c]}"></span> ${c}</th>`).join("")}<th>⚑</th></tr></thead><tbody>${
    rows.map(([k, r]) => `<tr><td>${k}</td><td>${r.total}</td><td>${r.tennis}</td><td>${r.hybrid}</td><td>${r.pickleball}</td><td>${r.padel}</td><td>${r.removed}</td><td>${r.flagged}</td></tr>`).join("")}</tbody>` : '<tbody><tr><td class="empty">No data yet.</td></tr></tbody>';

  // ---- downloads ---------------------------------------------------------------
  const files = [["courts.geojson", "Every court footprint with class history (GeoJSON)"], ["summary_by_county.csv", "Summary by county (CSV)"],
    ["summary_by_state.csv", "Summary by state (CSV)"], ["transitions.csv", "Every detected change (CSV)"]];
  $("downloads").innerHTML = files.map(([f, l]) => `<a href="data/downloads/${f}" download title="${l}">${f}</a>`).join("") +
    '<br><small>Also in the repository: <code>data/output/courts.parquet</code> (GeoParquet) and the raw model output under <code>data/detections/raw/</code>.</small>';
})();
