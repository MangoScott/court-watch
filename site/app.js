/* Court Watch static map. Data comes from site/data/ (written by scripts/07_export.py):
   courts_points.geojson, sites.json, summary.json, chips/<site>/<year>.jpg. No backend. */
(async function () {
  const CLASSES = ["tennis", "hybrid", "pickleball", "padel", "removed", "unknown"];
  const COLORS = { tennis: "#2ecc71", hybrid: "#f5a623", pickleball: "#e94b3c", padel: "#9b59b6", removed: "#7f8c8d", unknown: "#4b5563" };

  const [summary, courts, sites] = await Promise.all([
    fetch("data/summary.json").then(r => r.json()).catch(() => ({})),
    fetch("data/courts_points.geojson").then(r => r.json()),
    fetch("data/sites.json").then(r => r.json()).catch(() => ({})),
  ]);

  // ---- headline -----------------------------------------------------------
  const $ = id => document.getElementById(id);
  const num = v => (v === undefined || v === null ? "–" : Number(v).toLocaleString());
  $("n-hybrid").textContent = num(summary.current_hybrid);
  $("n-tennis").textContent = num(summary.current_tennis);
  $("n-pickleball").textContent = num(summary.current_pickleball);
  $("n-pb-courts").textContent = num(summary.pickleball_courts);
  $("n-padel").textContent = num(summary.current_padel);
  $("n-removed").textContent = num(summary.current_removed);
  if (summary.demo) {
    const b = document.createElement("div");
    b.id = "demo-banner";
    b.textContent = "Demo with synthetic data. These are not real courts or real numbers; the Ohio run has not been published yet.";
    document.body.prepend(b);
  }
  if (summary.state) $("state").textContent = { OH: "Ohio" }[summary.state] || summary.state;
  $("fine").textContent = summary.generated_at
    ? `${num(summary.footprints)} court footprints at ${num(summary.n_sites)} sites. ${num(summary.needs_review)} flagged for review. Updated ${summary.generated_at.slice(0, 10)}.`
    : "";

  // ---- filters ------------------------------------------------------------
  const active = new Set(CLASSES);
  const cf = $("class-filters");
  for (const c of CLASSES) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="checkbox" checked data-class="${c}"><span class="sw" style="background:${COLORS[c]}"></span>${c}`;
    cf.appendChild(l);
  }
  cf.addEventListener("change", e => {
    const c = e.target.dataset.class;
    if (!c) return;
    e.target.checked ? active.add(c) : active.delete(c);
    applyFilter();
  });
  $("only-review").addEventListener("change", applyFilter);
  $("only-changed").addEventListener("change", applyFilter);

  function applyFilter() {
    const f = ["all", ["in", ["get", "current_class"], ["literal", [...active]]]];
    if ($("only-review").checked) f.push(["==", ["get", "needs_review"], true]);
    if ($("only-changed").checked) f.push([">", ["get", "n_transitions"], 0]);
    if (!map.getSource("courts")) return;
    map.getSource("courts").setData({ type: "FeatureCollection", features: courts.features.filter(ft => passes(ft.properties)) });
  }
  function passes(p) {
    if (!active.has(p.current_class)) return false;
    if ($("only-review").checked && !p.needs_review) return false;
    if ($("only-changed").checked && !(p.n_transitions > 0)) return false;
    return true;
  }

  // ---- map ----------------------------------------------------------------
  const bounds = courts.features.length ? courts.features.reduce((b, f) => {
    const [x, y] = f.geometry.coordinates; return [Math.min(b[0], x), Math.min(b[1], y), Math.max(b[2], x), Math.max(b[3], y)];
  }, [180, 90, -180, -90]) : [-84.8, 38.4, -80.5, 42.0];

  const map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/positron",
    bounds: [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
    fitBoundsOptions: { padding: 40, maxZoom: 15 },
    attributionControl: true,
  });
  map.addControl(new maplibregl.NavigationControl(), "top-right");

  map.on("load", () => {
    map.addSource("courts", { type: "geojson", data: courts, cluster: true, clusterRadius: 40, clusterMaxZoom: 13 });
    map.addLayer({
      id: "clusters", type: "circle", source: "courts", filter: ["has", "point_count"],
      paint: { "circle-color": "#3b4c66", "circle-stroke-color": "#fff", "circle-stroke-width": 1,
        "circle-radius": ["step", ["get", "point_count"], 14, 20, 18, 100, 24, 500, 30] },
    });
    map.addLayer({
      id: "cluster-count", type: "symbol", source: "courts", filter: ["has", "point_count"],
      layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": 12 }, paint: { "text-color": "#fff" },
    });
    map.addLayer({
      id: "courts", type: "circle", source: "courts", filter: ["!", ["has", "point_count"]],
      paint: {
        "circle-color": ["match", ["get", "current_class"], ...CLASSES.flatMap(c => [c, COLORS[c]]), "#4b5563"],
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 4, 14, 8, 18, 12],
        "circle-stroke-color": ["case", ["get", "needs_review"], "#f5a623", "#fff"],
        "circle-stroke-width": ["case", ["get", "needs_review"], 2, 1],
      },
    });
    map.on("click", "clusters", e => {
      const f = map.queryRenderedFeatures(e.point, { layers: ["clusters"] })[0];
      map.getSource("courts").getClusterExpansionZoom(f.properties.cluster_id).then(z =>
        map.easeTo({ center: f.geometry.coordinates, zoom: z }));
    });
    map.on("click", "courts", e => showDetail(e.features[0]));
    for (const l of ["clusters", "courts"]) {
      map.on("mouseenter", l, () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", l, () => (map.getCanvas().style.cursor = ""));
    }
    const hash = location.hash.slice(1);
    if (hash) { const f = courts.features.find(x => x.properties.court_id === hash); if (f) { showDetail(f); map.jumpTo({ center: f.geometry.coordinates, zoom: 17 }); } }
  });

  // ---- detail panel -------------------------------------------------------
  let current = null;
  function showDetail(f) {
    const p = f.properties;
    const site = sites[p.site_id];
    current = { p, site };
    const history = typeof p.history === "string" ? JSON.parse(p.history) : p.history || [];
    const transitions = typeof p.transitions === "string" ? JSON.parse(p.transitions) : p.transitions || [];
    $("detail").hidden = false;
    $("d-title").textContent = p.court_id;
    $("d-class").innerHTML = `Now <span class="tag ${p.current_class}">${p.current_class}</span>` +
      (p.current_class === "pickleball" && p.current_n_courts > 1 ? ` (${p.current_n_courts} courts)` : "") +
      ` <span class="fine">confidence ${Number(p.current_confidence).toFixed(2)}${p.needs_review ? " · flagged for review" : ""}${p.n_overrides ? " · reviewed" : ""}</span>`;
    $("d-history").innerHTML = history.map(h =>
      `<div class="hist ${h.inferred ? "inferred" : ""}" title="${h.imagery_date || ""}${h.inferred ? " (no detection: inferred)" : ""}${h.interpolated ? " (interpolated)" : ""}">${h.year}<span class="tag ${h.class}">${h.class}${h.class === "pickleball" && h.n_courts > 1 ? " ×" + h.n_courts : ""}</span></div>`).join("");
    $("d-transitions").innerHTML = transitions.length
      ? `<h2>Changes</h2><ul>${transitions.map(t => `<li>${t.year_from}→${t.year_to}: <span class="tag ${t.from}">${t.from}</span> → <span class="tag ${t.to}">${t.to}</span>${t.flagged ? ` <span class="flag">⚠ ${t.flag_reasons}</span>` : ""}</li>`).join("")}</ul>`
      : `<h2>Changes</h2><p class="fine">No change across ${history.length} imagery years.</p>`;
    $("d-meta").innerHTML = `Site ${p.site_id}${p.county ? ", " + p.county + " County" : ""}. <a href="https://www.google.com/maps/@${f.geometry.coordinates[1]},${f.geometry.coordinates[0]},19z/data=!3m1!1e3" target="_blank" rel="noopener">Open in Google Maps</a>`;
    history.location = f.geometry.coordinates;
    location.hash = p.court_id;
    setupSlider(site, history);
    $("panel").scrollTo({ top: $("detail").offsetTop - 10, behavior: "smooth" });
  }
  $("close-detail").onclick = () => { $("detail").hidden = true; location.hash = ""; };

  function setupSlider(site, history) {
    const years = site ? site.years : history.map(h => h.year);
    const before = $("before-year"), after = $("after-year");
    before.innerHTML = after.innerHTML = years.map(y => `<option>${y}</option>`).join("");
    before.value = years[0]; after.value = years[years.length - 1];
    before.onchange = after.onchange = render;
    $("range").oninput = () => setSplit($("range").value);
    setSplit(50);
    render();
    function render() {
      const src = y => (site && site.images[y]) || "";
      $("img-before").src = src(before.value);
      $("img-after").src = src(after.value);
      const d = y => (site && site.imagery_dates && site.imagery_dates[y]) || y;
      $("date-before").textContent = `${before.value} · ${d(before.value)}`;
      $("date-after").textContent = `${after.value} · ${d(after.value)}`;
      drawOverlay(site, history, +before.value, +after.value);
    }
  }
  function setSplit(pct) {
    $("img-after").style.clipPath = `inset(0 0 0 ${pct}%)`;
    $("handle").style.left = pct + "%";
  }
  function drawOverlay(site, history, yBefore, yAfter) {
    const cv = $("overlay"); const ctx = cv.getContext("2d");
    cv.width = cv.height = 512; ctx.clearRect(0, 0, 512, 512);
    if (!current) return;
    const court = site && site.courts.find(c => c.court_id === current.p.court_id);
    const obbFor = y => { const h = court && court.history.find(x => x.year === y); return h ? h.obb : court && court.obb; };
    const draw = (obb, color, dashed) => {
      if (!obb) return;
      ctx.beginPath(); obb.forEach(([x, y], i) => (i ? ctx.lineTo(x * 512, y * 512) : ctx.moveTo(x * 512, y * 512))); ctx.closePath();
      ctx.setLineDash(dashed ? [6, 4] : []); ctx.lineWidth = 2.5; ctx.strokeStyle = color; ctx.stroke();
    };
    const cls = y => { const h = history.find(x => x.year === y); return h ? h.class : "unknown"; };
    draw(obbFor(yBefore), COLORS[cls(yBefore)] || "#fff", true);
    draw(obbFor(yAfter), COLORS[cls(yAfter)] || "#fff", false);
  }
})();
