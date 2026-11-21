"use strict";
// Mooz: one page over the server's JSON routes (gui/server.py). No framework: the views are
// template strings, the clicks are delegated through data-act attributes. Every value that
// comes from a file name, a settings file or the run goes into a template through esc().

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  projects: [], job: null, lines: 0, log: [],
  route: { name: "home" },
  project: null,            // the shown project's detail
  sel: {},                  // per project: the ticked takes and profiles
  settings: null, edits: {}, run: null, profileEdit: null, rawOpen: false,
  map: null,                // the shown take's files along the pipeline
  cur: null, mapRow: null,  // the pad whose details are shown, and the row the arrow keys are on
  box: null,                // the graph's chosen box, when it is a step with no file of its own
  view: "grid",             // the take's page: the grid, the graph, or the blind comparison
  blind: null,              // the blind comparison of the shown take: its two profiles, which is A, the choice
  lined: true, extrasOpen: false, help: null,
  draft: { name: "", folder: "", profiles: new Set(["default"]) }, browse: null, preview: null, browsing: false,
  logOpen: false, finished: -1,
  dismissed: Number(sessionStorage.getItem("dismissed") || 0),
};

// ---- helpers ---------------------------------------------------------------------------------

async function api(method, url, body) {
  const res = await fetch(url, {
    method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({ error: `The app answered ${res.status}.` }));
  if (!res.ok) throw new Error(data.error || `The app answered ${res.status}.`);
  return data;
}

let toastTimer;
function toast(message, bad = false) {
  const t = $("#toast");
  t.textContent = message;
  t.className = "show" + (bad ? " bad" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = ""), bad ? 7000 : 2600);
}
const fail = (e) => toast(e.message, true);

const minus = (x, digits = 1) => Number(x).toFixed(digits).replace("-", "−");
function clock(seconds) {
  if (seconds == null) return "";
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
function span(seconds) {
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
  return `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")} min`;
}
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const fileUrl = (pid, root, rel) => `/files/${pid}/${root}/${rel.split("/").map(encodeURIComponent).join("/")}`;
const takeUrl = (pid, take, profile) => `#/p/${pid}/take/${encodeURIComponent(take)}${profile ? `/${encodeURIComponent(profile)}` : ""}`;
const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);
const num = (v) => (Number.isInteger(v) ? String(v) : String(Math.round(v * 100) / 100)).replace("-", "−");

// The job's bar: the logo's idea. Ahead of the playhead the take is uneven raw bars; behind
// it, a clean wave in the rings' colours.
function wave(frac, seed) {
  const W = 720, H = 60, mid = H / 2, n = 90, cut = Math.max(0, Math.min(1, frac)) * W;
  let r = seed >>> 0 || 1, raw = "", clean = "";
  const rand = () => ((r = (Math.imul(r, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < n; i++) {
    const x = (i + 0.5) * (W / n), a = (0.18 + 0.82 * rand() * rand()) * mid * 0.92;
    if (x > cut + 2) raw += `M${x.toFixed(1)} ${(mid - a).toFixed(1)}V${(mid + a).toFixed(1)}`;
  }
  for (let x = 0; x <= cut; x += 3) clean += `${x ? "L" : "M"}${x} ${(mid - Math.sin((x / W) * Math.PI * 14) * mid * 0.62).toFixed(1)}`;
  const pct = Math.round(frac * 100);
  return `<svg class="wave" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="progressbar" aria-valuemin="0" aria-valuemax="100"
    aria-valuenow="${pct}" aria-label="Job progress"><path class="raw" d="${raw}"/><path class="clean" d="${clean}"/>
    <path class="head-line" d="M${cut.toFixed(1)} 2V${H - 2}"/></svg>`;
}
const seedOf = (text) => [...String(text)].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);

// ---- the rail --------------------------------------------------------------------------------

function renderRail() {
  const busy = S.job && S.job.state === "running" ? S.job.project.id : null;
  $("#projects").innerHTML = S.projects.length
    ? S.projects.map((p) => `<a href="#/p/${p.id}" class="${p.exists ? "" : "gone"} ${p.id === busy ? "busy" : ""}"
        ${S.route.id === p.id ? 'aria-current="page"' : ""} title="${esc(p.folder)}">${esc(p.name)}</a>`).join("")
    : `<span class="muted">No projects yet.</span>`;
}

// ---- the job ---------------------------------------------------------------------------------

function jobHeadline(j) {
  const n = j.units.length, verb = j.separate_only ? "Separating" : "Remastering";
  if (j.state === "running") return `${verb} ${j.project.name}`;  // set as text, not as HTML
  const failed = j.units.filter((u) => u.state === "failed").length;
  const kept = j.units.filter((u) => u.kept).length;  // already mastered with these very settings: not made again
  if (j.state === "done" && kept === n) return `Nothing to do: ${n === 1 ? "the master is" : `all ${n} masters are`} up to date`;
  if (j.state === "done") return `Done: ${plural(n - kept, j.separate_only ? "take" : "master")} in ${span(j.elapsed)}${kept ? `, ${kept} already up to date` : ""}`;
  if (j.state === "cancelled") return "Cancelled";
  return failed && failed < n ? `Finished, ${failed} of ${n} failed` : "The job failed";
}

function jobNow(j) {
  if (j.state !== "running") return esc(j.message || "");
  if (j.separation) {
    const s = j.separation;
    const step = s.n > 20 ? `${Math.round((100 * s.i) / s.n)} %` : s.n ? `model ${s.i + 1} of ${s.n}` : "starting";
    return `Separating <strong>${esc(s.track)}</strong>: ${step}${s.model ? ` <span class="muted">${esc(s.model)}</span>` : ""}`;
  }
  if (j.ahead) return `Checking the stems of every take first: <strong>${esc(j.ahead.take)}</strong> (${j.ahead.i + 1} of ${j.ahead.n})`;
  const u = j.current == null ? null : j.units[j.current];
  if (!u) return "Starting";
  return `<strong>${esc(u.take)}</strong> <span class="muted">${esc(u.profile)}</span> ${esc(u.stage)}${u.detail ? `: ${esc(u.detail)}` : ""}`;
}

function renderJob() {
  const el = $("#job"), j = S.job;
  if (!j || (j.state !== "running" && j.started === S.dismissed)) {
    el.innerHTML = "";
    el.dataset.key = "";
    return;
  }
  const key = String(j.started);
  if (el.dataset.key !== key) {  // a new job: its frame once, then only the parts that move
    el.dataset.key = key;
    el.innerHTML = `<div class="job-head"><h2 id="job-title"></h2><div class="row" id="job-buttons"></div></div>
      <div id="job-wave"></div>
      <div class="job-now"><span id="job-now"></span><span class="muted num" id="job-time"></span></div>
      <ul class="units" id="job-units"></ul>
      <details class="log" ${S.logOpen ? "open" : ""}><summary>What the run printed</summary><pre id="job-log"></pre></details>`;
    $("#job-log").textContent = S.log.join("\n");
  }
  const running = j.state === "running";
  $("#job-title").textContent = jobHeadline(j);
  $("#job-buttons").innerHTML = running
    ? `<span class="muted num">${j.units.filter((u) => u.state === "done" || u.state === "failed").length} of ${j.units.length} done</span>
       <button class="quiet small danger" data-act="cancel-job" type="button">Cancel job</button>`
    : `<a class="button quiet small" href="#/p/${j.project.id}">Open ${esc(j.project.name)}</a>
       <button class="quiet small" data-act="dismiss-job" type="button">Dismiss</button>`;
  $("#job-wave").innerHTML = wave(j.state === "done" ? 1 : j.frac, seedOf(key));
  $("#job-now").innerHTML = jobNow(j);
  $("#job-now").className = j.state === "failed" ? "error" : "";
  $("#job-time").textContent = running
    ? `${span(j.elapsed)} so far` + (j.eta != null ? `, about ${span(j.eta)} left` : "")
    : j.state === "done" ? "" : `after ${span(j.elapsed)}`;
  const list = $("#job-units"), top = list.scrollTop;
  list.innerHTML = j.units.map((u) => `<li class="${u.state}">
      <span class="dot" title="${u.state}"></span><span class="what">${esc(u.take)}</span><span class="muted what">${esc(u.profile)}</span>
      <span class="bar"><i style="width:${Math.round(u.frac * 100)}%"></i></span>
      <span class="secs num" ${u.kept ? 'title="Already mastered with these settings, recordings and code: not made again"' : ""}>${u.kept ? "up to date" : u.seconds != null ? span(u.seconds) : u.state === "running" ? `${Math.round(u.frac * 100)} %` : ""}</span>
      ${u.error ? `<span class="unit-error">${esc(u.error)}</span>` : ""}</li>`).join("");
  list.scrollTop = top;
}

function appendLog(lines) {
  if (!lines.length) return;
  S.log.push(...lines);
  if (S.log.length > 2000) S.log.splice(0, S.log.length - 2000);
  const pre = $("#job-log");
  if (!pre) return;
  const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
  pre.textContent = S.log.join("\n");
  if (atEnd) pre.scrollTop = pre.scrollHeight;
}

let pollTimer;
async function poll() {
  clearTimeout(pollTimer);
  try {
    const st = await api("GET", `/api/state?since=${S.lines}`);
    const before = JSON.stringify(S.projects);
    S.projects = st.projects;
    S.home = st.home;
    const j = st.job;
    if (j && (!S.job || S.job.started !== j.started)) {  // another job: its log from the start
      S.log = [];
      S.finished = -1;
      if (S.lines) { S.lines = 0; S.job = j; return poll(); }
    }
    S.job = j;
    if (j) { S.lines = j.lines; }
    if (before !== JSON.stringify(S.projects) || j) renderRail();
    renderJob();
    markWorking();
    if (j) {
      appendLog(j.log);
      const finished = j.units.filter((u) => u.state === "done" || u.state === "failed").length + (j.state === "running" ? 0 : 1);
      if (finished !== S.finished) {  // a take ended: its figures are on the project's page
        const first = S.finished === -1;
        S.finished = finished;
        if (!first && S.route.id === j.project.id && S.route.name === "project") loadProject(true);
        if (!first && S.route.id === j.project.id && S.route.name === "take") loadProject(true).then(() => loadMap(true));
        if (!first) updateActions();
      }
    }
  } catch (e) { /* the app is not answering: keep trying, quietly */ }
  pollTimer = setTimeout(poll, S.job && S.job.state === "running" ? 1000 : 4000);
}

// ---- views -----------------------------------------------------------------------------------

function parseRoute() {
  let parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  try { parts = parts.map(decodeURIComponent); } catch (e) { return { name: "home" }; }
  if (parts[0] === "new") return { name: "new" };
  if (parts[0] === "help") return { name: "help" };
  if (parts[0] === "p" && parts[1]) {
    if (parts[2] === "take" && parts[3]) return { name: "take", id: parts[1], take: parts[3], profile: parts[4] || "" };
    return { name: parts[2] === "settings" ? "settings" : "project", id: parts[1] };
  }
  return { name: "home" };
}

async function navigate() {
  S.route = parseRoute();
  renderRail();
  const r = S.route;
  if (r.name === "home") return renderHome();
  if (r.name === "new") return renderNew();
  if (r.name === "help") return renderHelp();
  if (!S.project || S.project.id !== r.id) { S.project = null; $("#view").innerHTML = `<p class="muted">Looking at the recordings</p>`; }
  if (r.name === "settings") { S.settings = null; S.edits = {}; S.run = null; S.profileEdit = null; }
  if (r.name !== "take" || !S.map || S.map.take.name !== r.take) { S.map = null; S.cur = S.mapRow = S.box = null; blindEnd(); }
  await loadProject();
  if (r.name === "settings") await loadSettings();
  if (r.name === "take") await loadMap();
}

function renderHome() {
  if (S.projects.length === 1 && !S.homeSeen) { S.homeSeen = true; location.hash = `#/p/${S.projects[0].id}`; return; }
  S.homeSeen = true;
  $("#view").innerHTML = `<div class="welcome"><h1>From the room to a master</h1>
    <p>A project is a folder of rehearsal recordings and the settings they are mastered with.
    ${S.projects.length ? "Choose one on the left, or start another." : "Start with the folder your recorder's files are in."}</p>
    <a class="button" href="#/new">New project</a></div>`;
}

// The guide is a page of the app's own (static/help.html), shown as it is.
async function renderHelp() {
  try {
    if (!S.help) S.help = await (await fetch("/static/help.html")).text();
    if (S.route.name === "help") $("#view").innerHTML = S.help;
  } catch (e) { $("#view").innerHTML = `<p class="error">The guide could not be read.</p>`; }
}

// -- a project's head, shared by its two pages
function projectHead(p, tab) {
  return `<div class="head"><div><h1>${esc(p.name)}</h1><div class="path">${esc(p.config)}</div></div>
      <button class="quiet small" data-act="remove-project" type="button">Remove from the list</button></div>
    <div class="tabs"><a href="#/p/${p.id}" ${tab === "project" ? 'aria-current="page"' : ""}>Takes</a>
      <a href="#/p/${p.id}/settings" ${tab === "settings" ? 'aria-current="page"' : ""}>Settings and profiles</a></div>`;
}

async function loadProject(quiet = false) {
  const id = S.route.id;
  try {
    const p = await api("GET", `/api/projects/${id}`);
    if (S.route.id !== id) return;
    S.project = p;
    if (!S.sel[id]) S.sel[id] = { takes: new Set(p.takes.map((t) => t.name)), profiles: new Set(p.run), ab: false, again: false };
    const sel = S.sel[id], names = new Set(p.takes.map((t) => t.name)), profs = new Set(p.profiles.map((x) => x.name));
    for (const t of [...sel.takes]) if (!names.has(t)) sel.takes.delete(t);
    for (const x of [...sel.profiles]) if (!profs.has(x)) sel.profiles.delete(x);
    if (S.route.name === "project") renderProject();
    if (S.route.name === "take" && p.error) $("#view").innerHTML = projectHead(p, "project") + `<div class="notice bad">${esc(p.error)}</div>`;
  } catch (e) {
    if (!quiet) $("#view").innerHTML = `<p class="error">${esc(e.message)}</p><a class="button quiet" href="#/">Back</a>`;
  }
}

function figureLinks(p, take, r) {
  const f = r.files;
  return `${f.figure ? `<a href="${esc(fileUrl(p.id, take.root, f.figure))}" target="_blank" rel="noopener">Mix figures</a>` : ""}
    ${f.tracks_figure ? ` <a href="${esc(fileUrl(p.id, take.root, f.tracks_figure))}" target="_blank" rel="noopener">Stem figures</a>` : ""}`;
}

// the recording itself, from the moment its masters start at (their trim)
function originalButton(p, t) {
  if (!t.room_file) return "";
  const trim = Object.values(t.results).map((r) => r.trim_s).find((x) => x != null) || 0;
  return `<button class="quiet play" type="button" data-act="play" data-take="${esc(`${p.id}/${t.name}`)}" data-offset="${trim}"
    data-src="${esc(fileUrl(p.id, t.root, t.room_file))}" data-label="${esc(t.title)}, as recorded" title="The recording itself">Original</button>`;
}

function resultCell(p, take, profile) {
  const r = take.results[profile];
  if (!r) return `<td class="result none" title="Not mastered with ${esc(profile)} yet">—</td>`;
  const f = r.files, label = `${take.title} (${profile})`, key = esc(`${p.id}/${take.name}`);
  const src = f.mp3 || f.wav, ab = f.ab_mp3 || f.ab_wav, mix = f.mix_mp3 || f.mix_wav;
  const when = r.when ? new Date(r.when * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
  const hot = r.limited_pct > 5;
  return `<td class="result">
    <div class="top">${r.lufs != null ? `<span title="Mastered ${esc(when)}"><span class="lufs num">${minus(r.lufs)}</span> <span class="more">LUFS</span></span>` : ""}
      ${src ? `<button class="quiet play" type="button" data-act="play" data-take="${key}" data-src="${esc(fileUrl(p.id, take.root, src))}" data-label="${esc(label)}">Play</button>` : ""}
      ${ab ? `<button class="quiet play" type="button" data-act="play" data-take="${key}" data-src="${esc(fileUrl(p.id, take.root, ab))}" data-label="${esc(label)}, original and master in turns" title="Original and master in turns, at the same loudness">A/B</button>` : ""}
      ${mix ? `<button class="quiet play" type="button" data-act="play" data-take="${key}" data-src="${esc(fileUrl(p.id, take.root, mix))}" data-label="${esc(label)}, the mix before the master"
        title="The mix before the master step, as a file: nothing clipped or limited${r.mix_lufs != null ? `, ${minus(r.mix_lufs)} LUFS` : ""}. Quieter than the master by what that step takes off the peaks">Mix</button>` : ""}</div>
    ${r.lufs != null ? `<div class="more num"><span title="Loudness range: the master's, of the recording's">range ${minus(r.lra_lu)} of ${minus(r.original_lra_lu)} LU</span>,
        <span class="${hot ? "warn" : ""}" title="${hot ? "The limiter works more than 5 % of the time: the master is pushed hard." : "How much of the time the limiter works."}">limiter ${minus(r.limited_pct, 0)} %</span></div>` : ""}
    <div class="more"><a href="${takeUrl(p.id, take.name, profile)}" title="Every stage of this take, to listen to">Stages</a>
      ${figureLinks(p, take, r)}</div></td>`;
}

function renderProject() {
  const p = S.project, sel = S.sel[p.id];
  if (p.error) {
    $("#view").innerHTML = projectHead(p, "project") + `<div class="notice bad">${esc(p.error)}</div>`;
    return;
  }
  // a column for every profile that has masters, or is ticked to run
  const cols = p.profiles.filter((x) => sel.profiles.has(x.name) || p.takes.some((t) => t.results[x.name]));
  const rows = p.takes.map((t) => `<tr class="take">
      <td><input type="checkbox" data-act="tick-take" data-take="${esc(t.name)}" ${sel.takes.has(t.name) ? "checked" : ""} aria-label="Choose ${esc(t.title)}"></td>
      <td><a class="take-title" href="${takeUrl(p.id, t.name)}" title="The take along the pipeline: its stems and every stage">${esc(t.title)}</a>
        <div class="take-sub num">${esc(t.date)}${t.title !== t.name ? `  ${esc(t.name)}` : ""}</div></td>
      <td><div class="tracks">${t.tracks.map((x) => `<span class="track">${esc(x)}</span>`).join("")}</div>
        <div class="orig">${originalButton(p, t)}</div></td>
      <td class="num">${clock(t.seconds)}</td>
      ${cols.map((c) => resultCell(p, t, c.name)).join("")}</tr>`).join("");
  $("#view").innerHTML = projectHead(p, "project") + (p.takes.length ? `
    <div class="table-wrap"><table>
      <thead><tr><th><input type="checkbox" data-act="tick-all" ${sel.takes.size === p.takes.length ? "checked" : ""} aria-label="Choose every take"></th>
        <th>Take</th><th>Tracks</th><th>Length</th>
        ${cols.map((c) => `<th title="${esc(c.description)}">${esc(c.name)}<br><span class="num">${esc(c.output_dir || "")}</span></th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody></table></div>
    ${heardHtml(p)}
    <div class="actions">
      <div><div class="muted" style="font-size:.87rem;margin-bottom:.3rem">Profiles to master with</div>
        <div class="chips">${p.profiles.map((x) => `<label class="chip" title="${esc(x.error || x.description)}">
          <input type="checkbox" data-act="tick-profile" data-profile="${esc(x.name)}" ${sel.profiles.has(x.name) ? "checked" : ""} ${x.error ? "disabled" : ""}>${esc(x.name)}</label>`).join("")}</div></div>
      <div class="go">
        <label title="Original and master in turns every 10 s, at the same loudness"><input type="checkbox" data-act="tick-ab" ${sel.ab ? "checked" : ""}> A/B files too</label>
        <label title="A take already mastered with these very settings is left as it is, so a job that stopped half-way goes on where it stopped. Tick this to make those again too."><input type="checkbox" data-act="tick-again" ${sel.again ? "checked" : ""}> Also what is up to date</label>
        <button class="quiet" type="button" data-act="separate" id="separate" title="Only split the takes into stems (the slow part, on the GPU) and keep them for later">Separate only</button>
        <button type="button" data-act="remaster" id="remaster"></button>
      </div>
    </div>` : `<div class="notice">No recordings found in this project's folders. Put the recorder's files next to
      <span class="num">${esc(p.config)}</span>, or change <code>[input] paths</code> under Settings and profiles.</div>`);
  updateActions();
}

// what the blind comparisons on the takes' pages have said so far (listening.json)
function heardHtml(p) {
  const h = p.listening || { verdicts: [], pairs: [] };
  if (h.error) return `<p class="error">${esc(h.error)}</p>`;
  if (!h.verdicts.length) return "";
  const title = (name) => (p.takes.find((t) => t.name === name) || {}).title || name;
  return `<div class="heard-box"><h3>What was heard</h3>
    ${h.pairs.length ? h.pairs.map((x) => `<p>${tallyHtml(x)}</p>`).join("")
      : `<p class="muted">Every verdict was on a master that has been made again since.</p>`}
    <details><summary>Every verdict (${h.verdicts.length})</summary>
      <ul class="heard">${h.verdicts.map((v) => verdictHtml(v, title(v.take))).join("")}</ul></details>
    <p class="muted">From comparing two profiles blind on a take's page. Kept in <span class="num">${esc(h.file)}</span></p></div>`;
}

function updateActions() {
  const b = $("#remaster"), one = $("#remaster-one"), p = S.project, busy = S.job && S.job.state === "running";
  if (one && S.map) {
    one.disabled = busy;
    one.textContent = busy ? "A job is running" : `${S.map.mastered ? "Remaster" : "Master"} this take with ${S.map.profile}`;
  }
  if (!b || !p) return;
  const sel = S.sel[p.id];
  const n = sel.takes.size, m = sel.profiles.size;
  b.disabled = $("#separate").disabled = busy || !n;
  if (!m) b.disabled = true;
  b.textContent = busy ? "A job is running" : !n ? "Choose takes" : !m ? "Choose a profile"
    : `Remaster ${plural(n, "take")}${m > 1 ? ` with ${m} profiles` : ""}`;
}

// -- a take along the pipeline: its recording, its stems, every stage file, the mix and the master

const PAD_ICON = `<svg viewBox="0 0 10 10" aria-hidden="true"><path class="tri" d="M3 1.6v6.8L8.6 5z"/><path class="bars" d="M2.6 1.8h1.7v6.4H2.6zM5.7 1.8h1.7v6.4H5.7z"/></svg>`;
// the run's stages (progress.py) on the page's columns, for showing where a running take is
const STAGE_COLUMN = { read: "recording", prepare: "recording", separate: "separated", profile: "separated", character: "separated",
  stems: "chain", drum_bus: "drum_bus", blend: "premix", bus: "mix", master: "master", write: "master", report: "master" };

const nodeById = (id) => (S.map && id ? [...S.map.nodes, ...S.map.extras].find((n) => n.id === id) : null) || null;
const nodeSrc = (n) => fileUrl(S.project.id, S.map.take.root, n.file);
const takeKey = () => `${S.project.id}/${S.map.take.name}`;

function nodeTitle(n) {
  const m = S.map;
  if (n.group) return `${n.group.toLowerCase()}, ${n.label}`;
  if (n.rows === "*") return n.col === "recording" ? `${n.label}, as recorded` : n.label;
  const stage = m.columns.find((c) => c.key === n.col).title.toLowerCase();
  const who = n.rows.length === 1 ? (m.rows.find((r) => r.key === n.rows[0]) || {}).label || n.label : n.label;
  return (who === stage ? who : `${who}, ${stage}`) + (n.main ? "" : ` (${n.label})`);
}

async function loadMap(quiet = false) {
  const r = S.route, p = S.project;
  if (!p || p.error || r.name !== "take") return;
  const t = p.takes.find((x) => x.name === r.take);
  const profile = r.profile || (p.profiles.find((x) => t && t.results[x.name]) || {}).name || p.run[0] || "default";
  try {
    const m = await api("GET", `/api/projects/${r.id}/takes/${encodeURIComponent(r.take)}/stages?profile=${encodeURIComponent(profile)}`);
    if (S.route !== r) return;
    S.map = m;
    if (!nodeById(S.cur)) S.cur = null;
    renderTake();
    // the same pad under another profile, or after a remaster: go on playing it there
    const n = nodeById(S.cur), a = $("#audio");
    if (n && P.take === takeKey() && !a.paused && !a.ended && nodeSrc(n) !== P.src) playNode(n);
    if (S.box && !boxById(S.box)) S.box = null;
  } catch (e) {
    if (!quiet) $("#view").innerHTML = projectHead(p, "project") + `<p class="error">${esc(e.message)}</p><a class="button quiet" href="#/p/${p.id}">All takes</a>`;
  }
}

function padHtml(n, word) {
  const round = n.main && !word && !n.short, title = nodeTitle(n);
  return `<button type="button" class="pad ${round ? "round" : "word"}${n.main && n.short ? " main" : ""}${n.unused ? " unused" : ""}" data-act="play-node" data-node="${esc(n.id)}" data-src="${esc(nodeSrc(n))}"
    title="${esc(cap(title))}: ${esc(n.what)}" ${round ? `aria-label="${esc(title)}"` : ""}>${round ? PAD_ICON : esc(n.short || n.label)}</button>`;
}

function renderTake() {
  const p = S.project, m = S.map, t = m.take;
  const rows = m.rows.length ? m.rows : [{ key: "*", label: "" }];
  const rowAt = Object.fromEntries(rows.map((r, i) => [r.key, i]));
  // a cell: the pads of one column that belong to the same rows
  const cells = new Map();
  for (const n of m.nodes) {
    const at = n.rows === "*" ? [0, rows.length - 1] : n.rows.map((k) => rowAt[k]).filter((i) => i != null);
    if (!at.length) continue;
    const a = Math.min(...at), b = Math.max(...at), key = `${n.col}/${a}/${b}`;
    if (!cells.has(key)) cells.set(key, { col: m.columns.findIndex((c) => c.key === n.col), a, b, wide: n.rows === "*" || n.rows.length > 1, nodes: [] });
    cells.get(key).nodes.push(n);
  }
  const reach = rows.map(() => 0);  // how far along the pipeline each row has files: its line ends there
  for (const c of cells.values()) for (let i = c.a; i <= c.b; i++) reach[i] = Math.max(reach[i], c.col);
  const tk = p.takes.find((x) => x.name === t.name), res = tk && tk.results[m.profile];
  const ch = (m.report || {}).character;
  const groups = {};
  for (const n of m.extras) (groups[n.group] ||= []).push(n);
  $("#view").innerHTML = projectHead(p, "project") + `
    <div class="take-head">
      <div><a class="back" href="#/p/${p.id}">All takes</a>
        <h2>${esc(t.title)}</h2>
        <div class="take-sub num">${esc(t.date)}, ${clock(t.seconds)}${ch && ch.style ? `, ${esc(ch.style)}${ch.why ? ` (${esc(ch.why)})` : ""}` : ""}${ch && ch.tempo_bpm ? `, ${Math.round(ch.tempo_bpm)} bpm` : ""}</div></div>
      <button class="quiet" type="button" data-act="remaster-one" id="remaster-one"></button>
    </div>
    <div class="tabs sub" aria-label="Profiles">${p.profiles.filter((x) => x.output_dir).map((x) => `<a href="${takeUrl(p.id, t.name, x.name)}"
      ${x.name === m.profile ? 'aria-current="page"' : ""} class="${tk && tk.results[x.name] ? "" : "none"}"
      title="${esc(tk && tk.results[x.name] ? x.description : `Not mastered with ${x.name} yet`)}">${esc(x.name)}</a>`).join("")}</div>
    ${res && res.lufs != null ? `<p class="take-figures num">${minus(res.lufs)} LUFS, range ${minus(res.lra_lu)} of ${minus(res.original_lra_lu)} LU,
      limiter ${minus(res.limited_pct, 0)} % of the time. ${figureLinks(p, tk, res)}</p>` : ""}
    ${!m.mastered ? `<div class="notice">Not mastered with ${esc(m.profile)} yet. The recording${m.rows.length ? " and the stems already separated" : ""} can be heard; the rest of the pipeline fills in after a master.</div>`
      : !m.stage_dir ? `<div class="notice">This master has no stage files${m.stages_setting === "off" ? ": Stage files is off in this profile's settings" : ""}. Turn them on and remaster the take to hear every stage.</div>` : ""}
    <div class="tabs sub views" role="tablist" aria-label="How the take is drawn">
      <button type="button" role="tab" data-act="view" data-view="grid" aria-selected="${S.view === "grid"}">Stage by stage</button>
      <button type="button" role="tab" data-act="view" data-view="graph" aria-selected="${S.view === "graph"}">Pipeline graph</button>
      <button type="button" role="tab" data-act="view" data-view="blind" aria-selected="${S.view === "blind"}">Compare blind</button></div>
    ${S.view === "blind" ? blindHtml() : (S.view === "graph" ? graphHtml(m) : `<div class="map-wrap"><div class="map" role="group" aria-label="The take along the pipeline">
      ${m.columns.map((c, i) => `<div class="col-head" data-col="${c.key}" style="grid-column:${i + 2}" title="${esc(c.about)}">${esc(c.title)}</div>`).join("")}
      ${rows.map((r, i) => `<div class="row-label" data-row="${esc(r.key)}" style="grid-row:${i + 2}">${esc(r.label)}</div>
        <i class="wire" data-row="${esc(r.key)}" style="grid-row:${i + 2};grid-column:2 / ${reach[i] + 3}"></i>`).join("")}
      ${[...cells.values()].map((c) => `<div class="cell${c.wide ? " wide" : ""}" style="grid-column:${c.col + 2};grid-row:${c.a + 2} / ${c.b + 3}">
        ${c.nodes.map((n) => padHtml(n, c.wide)).join("")}</div>`).join("")}
    </div></div>`) + `
    <label class="lined"><input type="checkbox" data-act="tick-lined" ${S.lined ? "checked" : ""}> Levels lined up</label>
    <span class="muted lined-help">Everything plays ${Math.abs(m.stage_gain_db)} dB under its real level, the way the stage files are written:
      the stages compare, and the raw drum peaks do not clip. Turn the volume up.</span>
    <div class="now" id="now"></div>
    ${m.extras.length ? `<details class="extras" ${S.extrasOpen ? "open" : ""}><summary>More files: every separation model's own output, and stems with no row of their own (${m.extras.length})</summary>
      ${Object.entries(groups).map(([g, ns]) => `<h4>${esc(g)}</h4><div class="row">${ns.map((n) => padHtml(n, true)).join("")}</div>`).join("")}</details>` : ""}`}`;
  if (S.view === "blind") blindShow();
  renderNow();
  markPlaying();
  markWorking();
  updateActions();
}

// -- the same take as a graph: a box per file, an arrow where one goes into the other

const GR = { dx: 138, dy: 40, w: 98, h: 28, top: 34, left: 6 };
// the graph's layers on the grid's columns, for showing where a running take is
const LAYER_COLUMN = ["recording", "separated", "separated", "separated", "cleaned", "chain", "drum_bus", "premix", "mix", "master"];
const boxById = (id) => (S.map && id ? S.map.graph.boxes.find((b) => b.id === id) : null) || null;
const chosenBox = () => boxById(S.box) || (S.cur && S.map ? S.map.graph.boxes.find((b) => b.play === S.cur) : null) || null;

function graphHtml(m) {
  const g = m.graph, at = {};
  for (const b of g.boxes) at[b.id] = { x: GR.left + b.layer * GR.dx, y: GR.top + b.lane * GR.dy };
  const W = GR.left * 2 + (g.layers.length - 1) * GR.dx + GR.w, H = GR.top + (Math.max(0, ...g.boxes.map((b) => b.lane)) + 1) * GR.dy;
  const arrows = g.arrows.map((a) => {
    const p = at[a.from], q = at[a.to], x1 = p.x + GR.w, y1 = p.y + GR.h / 2, x2 = q.x, y2 = q.y + GR.h / 2, bend = Math.max(16, (x2 - x1) / 2);
    const line = p.x === q.x ? `M${p.x + GR.w / 2} ${p.y + GR.h}V${q.y}`  // in the same layer: straight down
      : `M${x1} ${y1}C${x1 + bend} ${y1} ${x2 - bend} ${y2} ${x2} ${y2}`;
    return `<g class="garrow${a.unused ? " unused" : ""}" data-from="${esc(a.from)}" data-to="${esc(a.to)}">
      <path d="${line}"/>
      ${a.label ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 4}">${esc(a.label)}</text>` : ""}</g>`;
  }).join("");
  const boxes = g.boxes.map((b) => {
    const n = b.play ? nodeById(b.play) : null, p = at[b.id];
    const title = n ? `${cap(nodeTitle(n))}: ${n.what}` : b.what || b.label;
    return `<g class="gbox${b.step ? " step" : ""}${b.unused ? " unused" : ""}" transform="translate(${p.x} ${p.y})" tabindex="0" role="button"
      data-act="graph-box" data-box="${esc(b.id)}" ${n ? `data-src="${esc(nodeSrc(n))}"` : ""} aria-label="${esc(title)}"><title>${esc(title)}</title>
      <rect width="${GR.w}" height="${GR.h}" rx="${b.step ? 5 : 14}"/>
      ${n ? `<path class="tri" d="M11 8.5v11l9-5.5z"/><path class="bars" d="M10.5 8.5h3.2v11h-3.2zM16.3 8.5h3.2v11h-3.2z"/>` : ""}
      <text x="${n ? 27 : GR.w / 2}" y="${GR.h / 2 + 4.5}" ${n ? "" : 'text-anchor="middle"'}>${esc(b.label)}</text></g>`;
  }).join("");
  return `<div class="map-wrap"><svg class="graph" viewBox="0 0 ${W} ${H}" role="group" aria-label="The take's pipeline as a graph">
    ${g.layers.map((t, i) => `<text class="glayer" data-col="${LAYER_COLUMN[i]}" x="${GR.left + i * GR.dx}" y="16">${esc(t)}</text>`).join("")}${arrows}${boxes}</svg></div>`;
}

// the chosen box, and everything that goes into it or comes out of it
function markGraph() {
  const svg = $(".graph");
  if (!svg || !S.map) return;
  const g = S.map.graph, sel = chosenBox();
  const walk = (from, a, b) => { const seen = new Set(), todo = [from]; while (todo.length) { const x = todo.pop(); for (const e of g.arrows) if (e[a] === x && !seen.has(e[b])) { seen.add(e[b]); todo.push(e[b]); } } return seen; };
  const up = sel ? walk(sel.id, "to", "from") : new Set(), down = sel ? walk(sel.id, "from", "to") : new Set();
  svg.classList.toggle("picked", !!sel);
  svg.querySelectorAll(".gbox").forEach((el) => {
    const id = el.dataset.box;
    el.classList.toggle("cur", !!sel && id === sel.id);
    el.classList.toggle("off", !!sel && id !== sel.id && !up.has(id) && !down.has(id));
  });
  svg.querySelectorAll(".garrow").forEach((el) => {
    const a = el.dataset.from, b = el.dataset.to;
    el.classList.toggle("path", !!sel && (((up.has(a)) && (up.has(b) || b === sel.id)) || ((down.has(b)) && (down.has(a) || a === sel.id))));
  });
}

function chooseBox(b, play) {
  const n = b.play ? nodeById(b.play) : null;
  S.box = b.id;
  S.cur = n ? n.id : null;
  if (n && n.rows !== "*" && n.rows.length === 1) S.mapRow = n.rows[0];
  renderNow();
  if (n && play) playNode(n);
}

// the arrow keys in the graph: along an arrow (dx), or to the next box of the same layer (dy)
function nextBox(b, dx, dy) {
  const g = S.map.graph;
  if (dx) {
    const near = g.arrows.filter((a) => (dx > 0 ? a.from : a.to) === b.id).map((a) => boxById(dx > 0 ? a.to : a.from));
    return near.sort((p, q) => Math.abs(p.lane - b.lane) - Math.abs(q.lane - b.lane))[0] || null;
  }
  const layer = g.boxes.filter((x) => x.layer === b.layer).sort((p, q) => p.lane - q.lane);
  return layer[layer.indexOf(b) + dy] || null;
}

// -- what the take's report says of a pad's stage

const UNITS = [["_dbtp", " dBTP"], ["_dbfs", " dBFS"], ["_lufs", " LUFS"], ["_db", " dB"], ["_lu", " LU"], ["_pct", " %"], ["_hz", " Hz"],
  ["_ms", " ms"], ["_bpm", " bpm"], ["_s", " s"]];
function unitOf(key) {  // "avg_gr_db" is a number of dB called "avg gr"
  if (key === "lufs" || key.startsWith("lufs_")) return [`loudness${key.slice(4)}`, " LUFS"];
  for (const [end, unit] of UNITS) if (key.endsWith(end)) return [key.slice(0, -end.length), unit];
  return [key, ""];
}
const words = (key) => unitOf(key)[0].replace(/_/g, " ").replace(/\bgr\b/, "gain reduction").replace(/\bavg\b/, "average");
const NOT_THE_BUS = new Set(["file", "title", "date", "sample_rate", "channels", "duration_s", "original", "trim", "hum", "multitrack",
  "stem_profiles", "character", "tracks", "vocal_room", "finalize", "processed", "outputs", "ab_level_lufs"]);

function nodeFacts(n) {
  const r = S.map.report || {}, t = r.tracks || {}, mt = r.multitrack || {};
  const [, stem, variant] = n.id.split("/");
  if (n.group) return {};
  if (n.col === "recording") {
    if (n.main) return { as_recorded: r.original, trim: r.trim, hum: r.hum };
    return n.id === "recording/mixdown" ? { what_the_recorder_mixed: mt.recorder_mixdown } : { what_was_taken_from_it: (mt.close || {})[stem] };
  }
  if (n.col === "separated") {
    if (!variant) return { close_track_added: (mt.merged || {})[stem], as_separated: (r.stem_profiles || {})[stem] };
    return variant === "room" ? { in_the_merged_stem: (mt.merged || {})[stem] } : { against_the_room: ((mt.close || {})[variant] || {})[stem] };
  }
  if (n.col === "cleaned" || n.col === "chain") {
    if (variant) return {};
    const { profile, balance_gain_db, ...steps } = t[stem] || {};
    return steps;
  }
  if (n.col === "drum_bus") return t.drum_bus || {};
  if (n.col === "premix") {
    if (stem === "original") return { blend: t.blend };
    return { gain_into_the_mix_db: (t[stem] || {}).balance_gain_db, ...(stem === "vocals" ? { rider: t.rider } : {}), balance: t.balance };
  }
  if (n.col === "mix") return { ...Object.fromEntries(Object.entries(r).filter(([k]) => !NOT_THE_BUS.has(k))), rider: t.rider, loudness: t.remix };
  if (n.col === "master") return n.main ? { ...(r.finalize || {}), master: r.processed, recording: r.original } : { both_at_lufs: r.ab_level_lufs };
  return {};
}

function scalar(key, v) {
  if (typeof v === "number") return num(v) + unitOf(key)[1];
  if (typeof v === "boolean") return v ? "yes" : "no";
  return esc(v);
}

// a curve over frequency (an EQ's bands): bars around zero, and its ends in words
function bandsHtml(entries, unit) {
  const w = 9, H = 30, values = entries.map(([, x]) => x), top = Math.max(3, ...values.map(Math.abs));
  const signed = (x) => (x > 0 ? "+" : "") + num(x);
  const bars = entries.map(([k, x], i) => {
    const h = Math.max(0.6, (Math.abs(x) / top) * (H / 2 - 1));
    return `<rect class="${x < 0 ? "cut" : "boost"}" x="${i * w + 1}" y="${(x < 0 ? H / 2 : H / 2 - h).toFixed(1)}" width="${w - 2}" height="${h.toFixed(1)}"><title>${esc(k)} Hz: ${signed(x)}${unit}</title></rect>`;
  }).join("");
  return `<svg class="bands" viewBox="0 0 ${entries.length * w} ${H}" width="${entries.length * w * 1.5}" height="${H * 1.5}" role="img"
    aria-label="Curve over frequency">${bars}<path d="M0 ${H / 2}H${entries.length * w}"/></svg>
    <span class="muted">${esc(entries[0][0])} to ${esc(entries[entries.length - 1][0])} Hz, between ${signed(Math.min(...values))} and ${signed(Math.max(...values))}${unit}</span>`;
}

function valueHtml(key, v, depth = 0) {
  if (v == null) return "none";
  if (Array.isArray(v)) {
    if (v.every((x) => x === null || typeof x !== "object")) {
      return v.length === 2 && key.endsWith("_hz") ? `${num(v[0])} to ${num(v[1])} Hz` : v.map((x) => scalar(key, x)).join(", ");
    }
    return v.map((x) => `<div>${valueHtml(key, x, depth + 1)}</div>`).join("");
  }
  if (typeof v === "object") {
    const entries = Object.entries(v);
    if (entries.length > 2 && entries.every(([k, x]) => k.trim() !== "" && !Number.isNaN(Number(k)) && typeof x === "number")) return bandsHtml(entries, unitOf(key)[1]);
    if (depth < 2 && entries.some(([, x]) => x && typeof x === "object")) return factsHtml(v, depth + 1);
    return entries.map(([k, x]) => `${esc(words(k))} ${valueHtml(k, x, depth + 1)}`).join(", ");
  }
  return scalar(key, v);
}

function factsHtml(facts, depth = 0) {
  const rows = Object.entries(facts).filter(([, v]) => v != null && v !== "" && !(typeof v === "object" && !Object.keys(v).length));
  if (!rows.length) return "";
  return `<dl class="facts${depth ? " sub" : ""}">${rows.map(([k, v]) => `<dt>${esc(words(k))}</dt><dd>${valueHtml(k, v, depth)}</dd>`).join("")}</dl>`;
}

function renderNow() {
  const el = $("#now"), m = S.map;
  if (!el || !m) return;
  const n = nodeById(S.cur), step = n ? null : boxById(S.box);
  document.querySelectorAll(".pad[data-node]").forEach((b) => b.classList.toggle("cur", !!n && b.dataset.node === n.id));
  document.querySelectorAll(".map .wire").forEach((w) => w.classList.toggle("live", !!n && !n.group && (n.rows === "*" || n.rows.includes(w.dataset.row))));
  markGraph();
  if (step) {  // a separation: what it is, and each of its models' own outputs
    const groups = {};
    for (const x of m.extras) if (x.col === "models" && x.track === step.track) (groups[x.group] ||= []).push(x);
    el.innerHTML = `<h3>Separation of ${step.track === "room" ? "the room track" : esc(step.track)}</h3>
      <p>${esc(step.what)} <span class="muted">A stem that several models give is their weighted average. Each model's own output can be heard here.</span></p>
      ${Object.entries(groups).map(([g, ns]) => `<h4>${esc(g)}</h4><div class="row also">${ns.map((x) => padHtml(x, true)).join("")}</div>`).join("")
        || `<p class="muted">The models' own outputs were not kept for this take.</p>`}`;
    markPlaying();
    return;
  }
  if (!n) {
    el.innerHTML = S.view === "graph"
      ? `<p class="muted">Press a box to hear that point of the pipeline; what goes into it and what comes out of it stay lit. Going from box to box keeps
          the moment of the song. With a box in focus, the left and right arrow keys follow the arrows, up and down go through a layer.</p>`
      : `<p class="muted">Press a pad to hear that point of the pipeline. Going from pad to pad keeps the moment of the song.
          With a pad in focus, the arrow keys move along a stem's row or down a stage's column.</p>`;
    return;
  }
  const col = m.columns.find((c) => c.key === n.col), facts = factsHtml(nodeFacts(n));
  const also = n.group ? [] : m.nodes.filter((x) => x.id !== n.id && x.col === n.col && JSON.stringify(x.rows) === JSON.stringify(n.rows));
  el.innerHTML = `<h3>${esc(cap(nodeTitle(n)))}</h3>
    <p>${esc(cap(n.what))}.${col ? ` <span class="muted">${esc(col.about)}</span>` : ""}</p>
    ${also.length ? `<div class="row also"><span class="muted">At the same point:</span>${[n, ...also].map((x) => padHtml(x, true)).join("")}</div>` : ""}
    ${facts ? `<h4>What the take's report says here</h4>${facts}` : ""}
    <p class="path num">${esc(n.file)} <a href="${esc(nodeSrc(n))}" download>Download</a></p>`;
  markPlaying();
}

function selectNode(n) {
  S.cur = n.id;
  S.box = null;
  if (n.rows !== "*" && n.rows.length === 1) S.mapRow = n.rows[0];
  renderNow();
}

function playNode(n) {
  const m = S.map;
  play({ src: nodeSrc(n), label: `${m.take.title}: ${nodeTitle(n)} (${m.profile})`, take: takeKey(), offset: n.whole ? m.trim_s : 0,
    sync: n.sync, gainDb: S.lined ? m.stage_gain_db - n.level_db : 0 });
}

// the arrow keys: the next pad along the row (dx) or down the column (dy)
function nextPad(dx, dy) {
  const m = S.map, cur = nodeById(S.cur);
  if (!cur || cur.group) return null;
  const cols = m.columns.map((c) => c.key), rows = m.rows.map((r) => r.key);
  const at = (col, row) => m.nodes.find((n) => n.col === col && n.main && (n.rows === "*" || n.rows.includes(row))) || null;
  let c = cols.indexOf(cur.col), r = Math.max(0, rows.indexOf(S.mapRow));
  if (cur.rows !== "*" && !cur.rows.includes(rows[r])) r = rows.indexOf(cur.rows[0]);
  for (c += dx, r += dy; c >= 0 && c < cols.length && r >= 0 && r < rows.length; c += dx, r += dy) {
    const n = at(cols[c], rows[r]);
    if (n && n.id !== cur.id) { S.mapRow = rows[r]; return n; }
  }
  return null;
}

function markWorking() {
  document.querySelectorAll(".map .working, .graph .working").forEach((el) => el.classList.remove("working"));
  const j = S.job, m = S.map;
  if (!m || S.route.name !== "take" || !j || j.state !== "running" || j.current == null) return;
  const u = j.units[j.current];
  if (j.project.id !== S.route.id || u.take !== m.take.name || u.profile !== m.profile) return;
  const col = j.separation ? "separated" : STAGE_COLUMN[u.key];
  document.querySelectorAll(".map .col-head, .graph .glayer").forEach((el) => el.classList.toggle("working", el.dataset.col === col));
  if (u.key === "stems" && u.detail && !j.separation) {
    const stem = u.detail.replace(/, kept$/, "");  // "bass, kept": the chain an earlier run left is used as it is
    document.querySelectorAll(".map .row-label").forEach((el) => el.classList.toggle("working", el.dataset.row === stem));
  }
}

// ---- comparing blind -------------------------------------------------------------------------
// Two profiles' masters of the take as A and B, drawn by lot. Both are decoded whole and started
// on the same sample, so A and B switch with no gap and no slip, and a passage loops exactly;
// each is turned down to the quieter one's loudness. The choice goes to the project's
// listening.json (gui/listening.py); only then does the page say which was which.

const B = { key: "", bufs: {}, gains: {}, srcs: [], level: {}, lead: {}, dur: 0, playing: false, at: 0, t0: 0, loop: [null, null], timer: null };

const blindTake = () => (S.project && S.map ? S.project.takes.find((x) => x.name === S.map.take.name) : null);
// A side of the comparison is a profile's master ("pro"), or the mix it was made from ("pro:mix": <take>_mix, the
// stage before the master step at the one gain that puts its highest peak at the ceiling).
const MIX = ":mix";
const sideName = (id) => (id.endsWith(MIX) ? `${id.slice(0, -MIX.length)}, the mix` : id);
function sideOf(tk, id) {
  const mix = id.endsWith(MIX), r = tk.results[mix ? id.slice(0, -MIX.length) : id];
  if (!r) return null;
  const file = mix ? r.files.mix_wav || r.files.mix_mp3 : r.files.wav || r.files.mp3, lufs = mix ? r.mix_lufs : r.lufs;
  if (!file || lufs == null) return null;
  return { id, mix, file, lufs, lra: mix ? r.mix_lra_lu : r.lra_lu, plr: mix ? r.mix_plr_db : r.plr_db, limited: mix ? null : r.limited_pct,
    trim: r.trim_s || 0, when: r.when || 0 };
}
// the profiles this take can be compared under: those with a master, the last made first
function blindProfiles() {
  const tk = blindTake();
  if (!tk) return [];
  return Object.keys(tk.results).flatMap((x) => [x, x + MIX]).filter((x) => sideOf(tk, x))
    .sort((x, y) => sideOf(tk, y).when - sideOf(tk, x).when || x.length - y.length);  // a master, then its own mix
}
const blindLoop = () => (B.loop[0] != null && B.loop[1] != null && B.loop[1] - B.loop[0] >= 0.3 ? B.loop : null);

function blindPos() {
  if (!B.playing) return B.at;
  const p = B.at + audioContext.currentTime - B.t0, lp = blindLoop();
  return lp && p >= lp[1] ? lp[0] + ((p - lp[1]) % (lp[1] - lp[0])) : Math.min(p, B.dur);
}

function blindSilence() {
  for (const src of B.srcs) { src.onended = null; try { src.stop(); } catch (e) { /* not started */ } }
  B.srcs = [];
}

function blindPause() {
  if (!B.playing) return;
  B.at = blindPos();
  B.playing = false;
  blindSilence();
  blindMark();
}

function blindPlay(at = B.at) {
  const b = S.blind;
  if (!b || !b.order || !B.dur) return;
  const ctx = audioCtx(), lp = blindLoop();
  $("#audio").pause();  // the page's own player steps aside: one thing plays, and its bar names a profile
  $("#player").hidden = true;
  document.body.classList.remove("playing");
  blindSilence();
  if (lp && (at < lp[0] || at >= lp[1])) at = lp[0];
  if (at >= B.dur - 0.05) at = 0;
  const when = ctx.currentTime + 0.05;
  for (const x of b.order) {
    const src = ctx.createBufferSource(), late = B.lead[x];  // late: how far into the song this master starts after the other
    src.buffer = B.bufs[x];
    src.connect(B.gains[x]);
    if (lp) { src.loop = true; src.loopStart = Math.max(0, lp[0] - late); src.loopEnd = Math.max(0.1, lp[1] - late); }
    src.start(when + Math.max(0, late - at), Math.max(0, at - late));
    B.srcs.push(src);
  }
  B.srcs[0].onended = () => { if (B.playing && !blindLoop()) { B.playing = false; B.at = 0; B.srcs = []; blindMark(); } };
  B.at = at;
  B.t0 = when;
  B.playing = true;
  blindSide(b.on);
}

function blindSide(i) {
  const b = S.blind;
  if (!b || !b.order) return;
  b.on = i;
  b.order.forEach((x, k) => B.gains[x].gain.setTargetAtTime(k === i ? B.level[x] : 0, audioContext.currentTime, 0.006));
  blindMark();
}

function blindSeek(at) {
  B.at = Math.max(0, Math.min(B.dur, at));
  if (B.playing) blindPlay(B.at); else blindMark();
}

// the parts of the view that move: which side is on, the position, the loop
function blindMark() {
  const b = S.blind, seek = $("#blind-seek");
  if (!b || !seek) return;
  const pos = blindPos(), lp = blindLoop();
  document.querySelectorAll(".side").forEach((el) => {
    const mine = Number(el.dataset.side) === b.on;
    el.classList.toggle("on", mine);
    el.setAttribute("aria-pressed", String(mine));
  });
  if (document.activeElement !== seek || !seek.matches(":active")) seek.value = pos;
  $("#blind-time").textContent = `${clock(pos)} / ${clock(B.dur)}`;
  $("#blind-toggle").textContent = B.playing ? "Pause" : "Play";
  $("#blind-loop").textContent = lp ? `Looping ${clock(lp[0])} to ${clock(lp[1])}` : B.loop[0] != null ? `From ${clock(B.loop[0])}: now set where the loop ends`
    : B.loop[1] != null ? `To ${clock(B.loop[1])}: now set where the loop starts` : "";
}

// the comparison is over: silence, and the two decoded masters are let go
function blindEnd() {
  blindSilence();
  clearInterval(B.timer);
  Object.assign(B, { key: "", bufs: {}, gains: {}, level: {}, lead: {}, dur: 0, playing: false, at: 0, loop: [null, null], timer: null });
  S.blind = null;
}

async function blindStart() {
  const tk = blindTake(), b = S.blind;
  if (!tk || !b || b.pair[0] === b.pair[1]) return;
  const key = takeKey(), pair = [...b.pair];
  blindSilence();
  Object.assign(b, { order: null, loading: true, error: "", verdict: null, on: 0 });
  renderTake();
  try {
    const ctx = audioCtx();
    const bufs = await Promise.all(pair.map(async (x) => {
      if (B.key === key && B.bufs[x]) return B.bufs[x];
      const res = await fetch(fileUrl(S.project.id, tk.root, sideOf(tk, x).file));
      if (!res.ok) throw new Error(`The file of ${sideName(x)} could not be read.`);
      return ctx.decodeAudioData(await res.arrayBuffer());
    }));
    if (S.blind !== b || takeKey() !== key) return;  // the page moved on while they loaded
    const quiet = Math.min(...pair.map((x) => sideOf(tk, x).lufs)), first = Math.min(...pair.map((x) => sideOf(tk, x).trim));
    Object.assign(B, { key, bufs: {}, level: {}, lead: {}, at: B.key === key ? B.at : 0, playing: false });
    pair.forEach((x, i) => {
      B.bufs[x] = bufs[i];
      B.level[x] = 10 ** ((quiet - sideOf(tk, x).lufs) / 20);
      B.lead[x] = sideOf(tk, x).trim - first;
      if (!B.gains[x]) { B.gains[x] = ctx.createGain(); B.gains[x].connect(ctx.destination); }
      B.gains[x].gain.value = 0;
    });
    B.dur = Math.min(...pair.map((x) => B.bufs[x].duration + B.lead[x]));
    clearInterval(B.timer);
    B.timer = setInterval(() => { if (B.playing) blindMark(); }, 100);
    Object.assign(b, { loading: false, order: Math.random() < 0.5 ? pair : [pair[1], pair[0]] });
    renderTake();
    blindPlay();
  } catch (e) {
    if (S.blind === b) { Object.assign(b, { loading: false, error: e.message || "The two masters could not be decoded in this browser." }); renderTake(); }
  }
}

// what a pair of profiles' verdicts add up to, the one chosen more often first
function tallyHtml(pair) {
  const [x, y] = pair.wins[pair.profiles[0]] >= pair.wins[pair.profiles[1]] ? pair.profiles : [pair.profiles[1], pair.profiles[0]];
  const parts = [`<strong>${esc(sideName(x))}</strong> chosen in ${pair.wins[x]} of ${plural(pair.rounds, "round")}`, `<strong>${esc(sideName(y))}</strong> in ${pair.wins[y]}`];
  if (pair.same) parts.push(`no difference heard in ${pair.same}`);
  return `${parts.join(", ")} <span class="muted">(${plural(pair.takes, "take")})</span>`;
}

function verdictHtml(v, withTake) {
  const when = new Date(v.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  return `<li class="${v.stale ? "stale" : ""}"><span>${withTake ? `${esc(withTake)}: ` : ""}${v.choice === "same"
      ? `${withTake ? "no" : "No"} difference heard between ${esc(sideName(v.a))} and ${esc(sideName(v.b))}` : `<strong>${esc(sideName(v.choice))}</strong> over ${esc(sideName(v.choice === v.a ? v.b : v.a))}`}${
      v.loop_s ? `, on ${clock(v.loop_s[0])} to ${clock(v.loop_s[1])}` : ""}${v.note ? `: ${esc(v.note)}` : ""}
      <span class="muted">${esc(when)}${v.stale ? ", on masters replaced since: not counted" : ""}</span></span>
    <button class="quiet small" type="button" data-act="blind-take-back" data-id="${esc(v.id)}" title="Remove this verdict from listening.json">Take back</button></li>`;
}

function blindHtml() {
  const tk = blindTake(), ready = blindProfiles(), heard = S.project.listening || { verdicts: [], pairs: [] };
  if (ready.length < 2) {
    return `<div class="notice">Comparing needs this take mastered with two profiles${ready.length ? `; so far it has ${esc(ready[0])}` : ""}.
      Tick a second profile on the takes' page and master the take with it.</div>`;
  }
  if (!S.blind || S.blind.key !== takeKey() || S.blind.pair.some((x) => !ready.includes(x))) {
    blindEnd();
    S.blind = { key: takeKey(), pair: [ready[0], ready[1]], order: null, on: 0, verdict: null, note: "", loading: false, error: "" };
  }
  const b = S.blind, r = (x) => sideOf(tk, x);
  const pick = (i) => `<select data-act="blind-pick" data-i="${i}" aria-label="${i ? "Second" : "First"} profile">${ready.map((x) =>
    `<option value="${esc(x)}" ${x === b.pair[i] ? "selected" : ""}>${esc(sideName(x))}</option>`).join("")}</select>`;
  const name = (i) => (b.verdict ? esc(sideName(b.order[i])) : "");
  const mine = heard.verdicts.filter((v) => v.take === tk.name);
  const pair = heard.pairs.find((x) => x.profiles.includes(b.pair[0]) && x.profiles.includes(b.pair[1]));
  return `<div class="blind">
    <div class="row">Compare ${pick(0)} with ${pick(1)}
      ${b.order ? "" : `<button type="button" data-act="blind-start" ${b.loading || b.pair[0] === b.pair[1] ? "disabled" : ""}>${b.loading ? "Reading the two masters" : "Start"}</button>`}</div>
    ${b.error ? `<p class="error">${esc(b.error)}</p>` : ""}
    ${!b.order ? `<p class="muted">The two masters play in step as A and B, at the same loudness. Which is which is drawn by lot, and shown once you have chosen.</p>` : `
    <div class="sides" role="group" aria-label="Which master plays">
      ${[0, 1].map((i) => `<button type="button" class="side" data-act="blind-side" data-side="${i}" aria-pressed="false"><b>${"AB"[i]}</b><span>${name(i)}</span></button>`).join("")}</div>
    <div class="transport">
      <button type="button" class="quiet" data-act="blind-toggle" id="blind-toggle">Play</button>
      <input type="range" id="blind-seek" min="0" max="${B.dur.toFixed(1)}" step="0.1" value="0" aria-label="Moment of the song">
      <span class="num" id="blind-time"></span></div>
    <div class="row loop">
      <button type="button" class="quiet small" data-act="blind-loop" data-end="0">Loop from here</button>
      <button type="button" class="quiet small" data-act="blind-loop" data-end="1">Loop to here</button>
      <button type="button" class="quiet small" data-act="blind-loop-off">No loop</button>
      <span class="muted num" id="blind-loop"></span></div>
    <p class="muted keys">Keys: <kbd>A</kbd> and <kbd>B</kbd> choose the side, <kbd>X</kbd> the other one, <kbd>Space</kbd> plays and pauses, <kbd>[</kbd> and <kbd>]</kbd> set the loop.</p>
    ${b.verdict ? `<div class="reveal">
        <p><strong>A was ${esc(sideName(b.order[0]))}, B was ${esc(sideName(b.order[1]))}.</strong> ${b.verdict === "same" ? "You heard no difference." : `You chose ${esc(sideName(b.verdict))}.`}
          ${Math.abs(r(b.order[0]).lufs - r(b.order[1]).lufs) > 0.3 ? `<span class="muted">The louder one was turned down ${minus(Math.abs(r(b.order[0]).lufs - r(b.order[1]).lufs))} dB, so that both played equally loud.</span>` : ""}</p>
        <table class="figs num"><thead><tr><th></th><th>Loudness in the file</th><th>Range</th><th title="How far the highest peak stands over the loudness: the drum hits' punch">Peaks over the loudness</th><th>Limiter at work</th></tr></thead><tbody>
          ${b.order.map((x, i) => `<tr><td>${"AB"[i]}: ${esc(sideName(x))}</td><td>${minus(r(x).lufs)} LUFS</td><td>${minus(r(x).lra)} LU</td>
            <td>${r(x).plr != null ? `${minus(r(x).plr)} dB` : ""}</td><td>${r(x).limited == null ? "not limited" : `${minus(r(x).limited, 0)} % of the time`}</td></tr>`).join("")}</tbody></table>
        <button type="button" data-act="blind-again" title="A and B are drawn again; the song stops so that the change cannot be heard">Another round</button></div>`
      : `<div class="verdict">
        <input type="text" id="blind-note" maxlength="500" placeholder="What made the difference (optional)" value="${esc(b.note)}" aria-label="What made the difference">
        <div class="row"><button type="button" data-act="blind-choose" data-choice="0">A sounds better</button>
          <button type="button" data-act="blind-choose" data-choice="1">B sounds better</button>
          <button type="button" class="quiet" data-act="blind-choose" data-choice="same">I hear no difference</button></div></div>`}`}
    ${heard.error ? `<p class="error">${esc(heard.error)}</p>` : ""}
    ${pair ? `<p class="tally">So far, on the masters as they are now: ${tallyHtml(pair)}</p>` : ""}
    ${mine.length ? `<h4>Heard on this take</h4><ul class="heard">${mine.map((v) => verdictHtml(v)).join("")}</ul>` : ""}
  </div>`;
}

function blindShow() {
  const b = S.blind;
  if (b && b.order && b.verdict) document.querySelectorAll(".side").forEach((el) => el.classList.add("named"));
  blindMark();
}

// ---- listening -------------------------------------------------------------------------------
// One audio element for the whole page. What plays knows its take and where the song starts in
// its file (a whole recording starts before the trim, a stage file and a master at it), so
// going from one file of a take to another keeps the moment of the song.

const P = { src: "", take: "", offset: 0, sync: false, pos: 0, settling: false };
let audioContext = null, trimGain = null;

// the level a file is played at, apart from the player's own volume
function audioCtx() {
  if (!audioContext) {
    audioContext = new AudioContext();
    trimGain = audioContext.createGain();
    audioContext.createMediaElementSource($("#audio")).connect(trimGain).connect(audioContext.destination);
  }
  if (audioContext.state === "suspended") audioContext.resume();
  return audioContext;
}

function setGain(db) {
  const g = 10 ** (db / 20);
  try {
    audioCtx();
    trimGain.gain.value = g;
  } catch (e) { $("#audio").volume = Math.min(1, g); }
}

function play(o) {
  const a = $("#audio");
  if (o.src === P.src && !a.ended && !$("#player").hidden) {  // the one that is on: pause, or go on
    if (a.paused) a.play().catch(() => {}); else a.pause();
    return;
  }
  blindPause();
  if (o.take !== P.take) P.pos = 0;
  const at = o.sync ? Math.max(0, P.pos + o.offset) : 0;
  Object.assign(P, { src: o.src, take: o.take, offset: o.offset, sync: o.sync, settling: true });
  a.src = o.src;
  if (at > 0.05) {
    a.currentTime = at;
    a.addEventListener("loadedmetadata", () => { if (P.src === o.src && Math.abs(a.currentTime - at) > 0.3) a.currentTime = at; }, { once: true });
  }
  setGain(o.gainDb);
  $("#playing").textContent = o.label;
  $("#player").hidden = false;
  document.body.classList.add("playing");
  a.play().catch(() => {});
  markPlaying();
}

function markPlaying() {
  const a = $("#audio"), shown = !$("#player").hidden, on = shown && !a.paused && !a.ended;
  document.querySelectorAll("[data-src]").forEach((el) => {
    const mine = shown && el.dataset.src === P.src;
    el.classList.toggle("sel", mine);
    el.classList.toggle("on", mine && on);
    el.setAttribute("aria-pressed", String(mine && on));
  });
}

// -- settings and profiles

async function loadSettings() {
  const id = S.route.id;
  try {
    const s = await api("GET", `/api/projects/${id}/settings`);
    if (S.route.id !== id || S.route.name !== "settings") return;
    S.settings = s;
    S.run = new Set(s.run);
    renderSettings();
  } catch (e) {
    $("#view").innerHTML = (S.project ? projectHead(S.project, "settings") : "") + `<p class="error">${esc(e.message)}</p>`;
  }
}

// The same fields edit two files: the project's settings (what a field goes back to is the
// default) and a profile (it goes back to the project's value).
const PROJECT_FORM = { act: "knob", reset: "reset-knob", id: "k-", base: "Default", back: "Back to default",
  edits: () => S.edits, file: () => "rehearsal.toml" };
const PROFILE_FORM = { act: "pknob", reset: "reset-pknob", id: "pk-", base: "The project's", back: "Back to the project's",
  edits: () => S.profileEdit.edits, file: () => `presets/${S.profileEdit.name}.toml` };

function fieldHtml(f, form = PROJECT_FORM) {
  const edits = form.edits(), edited = f.key in edits, value = edited ? edits[f.key] : f.value, reset = edited && edits[f.key] === null;
  const shown = reset ? f.default : value;
  const id = `${form.id}${f.key}`, at = `id="${id}" data-act="${form.act}" data-key="${f.key}"`;
  let input;
  if (f.kind === "bool") input = `<input type="checkbox" ${at} ${shown ? "checked" : ""}>`;
  else if (f.kind === "choice") input = `<select ${at}>${f.choices.map((c) => `<option ${c === shown ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
  else if (f.kind === "text") input = `<input type="text" ${at} value="${esc(shown)}">`;
  else input = `<input type="number" ${at} value="${esc(shown)}" min="${f.min}" max="${f.max}" step="${f.step}">`;
  const def = f.kind === "bool" ? (f.default ? "on" : "off") : f.default === "" ? "empty" : f.kind === "number" ? minus(f.default, 2).replace(/\.?0+$/, "") : f.default;
  const isSet = (f.set && !reset) || (edited && !reset);
  return `<div class="field ${isSet ? "is-set" : ""}"><label for="${id}">${esc(f.label)}</label><span>${input}</span>
    <span class="help">${esc(f.help)} ${form.base}: ${esc(def)}.${isSet ? `<button class="quiet small" type="button" data-act="${form.reset}" data-key="${f.key}" title="Take ${f.key} out of ${esc(form.file())}">${form.back}</button>` : ""}</span></div>`;
}

const profileKind = (x) => (x.isNew ? "not saved yet" : x.own && x.builtin ? "built in, with this project's changes" : x.builtin ? "built in" : "this project's own");

function renderSettings() {
  const p = S.project, s = S.settings;
  if (!p || !s) return;
  const e = S.profileEdit;
  // what was typed in the two text boxes outlives a redraw of the same file
  const typed = $("#profile-text"), typedRaw = $("#raw-toml");
  if (typed && e && S.shownProfile === e) e.text = typed.value;
  if (typedRaw && S.shownSettings === s) s.toml = typedRaw.value;
  S.shownProfile = e;
  S.shownSettings = s;
  $("#view").innerHTML = projectHead(p, "settings") + `
    ${s.error ? `<div class="notice bad">rehearsal.toml does not load: ${esc(s.error)}. Fix it under "The whole file" below.</div>` : ""}
    <h2 style="margin-top:0">Profiles every take is mastered with</h2>
    <p class="muted">What is ticked on the takes page to begin with, and what a plain <code>mooz</code> runs.</p>
    <div class="chips">${p.profiles.map((x) => `<label class="chip" title="${esc(x.description)}"><input type="checkbox" data-act="tick-run" data-profile="${esc(x.name)}" ${S.run.has(x.name) ? "checked" : ""}>${esc(x.name)}</label>`).join("")}</div>

    ${s.groups.map((g) => `<h2>${esc(g.title)}</h2>${g.fields.map((f) => fieldHtml(f)).join("")}`).join("")}
    <p class="muted" style="margin-top:1rem">A dot marks what this project sets itself. Everything else, with its explanation, is in
      <span class="num">${esc(s.defaults_file)}</span>.</p>
    <div class="row" style="margin-top:1rem"><button type="button" data-act="save-settings" id="save-settings">Save settings</button>
      <span class="error" id="settings-error"></span></div>

    <h2>Profiles</h2>
    <p class="muted">A profile is a sound: a few settings on top of the project's, with a folder of its own for the masters.
      Every one can be edited. A built-in one that is changed here is kept as this project's version of it.</p>
    <div class="table-wrap"><table class="profiles"><tbody>${p.profiles.filter((x) => x.name !== "default").map((x) => `<tr>
      <td>${esc(x.name)}</td><td>${esc(x.error || x.description)}<div class="take-sub"><span class="num">${esc(x.output_dir || "")}</span>, ${profileKind(x)}</div></td>
      <td><button class="quiet small" type="button" data-act="open-profile" data-profile="${esc(x.name)}">Edit</button></td></tr>`).join("")}
    </tbody></table></div>
    <div class="row" style="margin-top:.9rem">
      <input type="text" id="new-profile" placeholder="name of a new profile" style="width:14rem" aria-label="Name of a new profile">
      <select id="new-profile-from" style="width:auto" aria-label="What it starts from"><option value="">starting empty</option>
        ${p.profiles.filter((x) => x.name !== "default").map((x) => `<option value="${esc(x.name)}">a copy of ${esc(x.name)}</option>`).join("")}</select>
      <button class="quiet" type="button" data-act="new-profile">Start profile</button></div>
    ${e ? `<div class="editor" id="profile-editor"><h3>${esc(e.name)} <span class="muted">${profileKind(e)}</span></h3>
      ${e.builtin && !e.own ? `<p class="muted">Saving writes this project's version to <span class="num">presets/${esc(e.name)}.toml</span>, next to rehearsal.toml,
        and the project uses it from then on. The built-in profile itself is not changed: other projects keep it, and so can this one, by going back to it.</p>` : ""}
      ${e.error ? `<div class="notice bad">This profile does not load: ${esc(e.error)}. Fix it in its file below.</div>` : ""}
      ${e.groups.map((g) => `<h4>${esc(g.title)}</h4>${g.fields.map((f) => fieldHtml(f, PROFILE_FORM)).join("")}`).join("")}
      ${e.groups.length ? `<p class="muted" style="margin-top:.8rem">A dot marks what this profile sets itself. Everything else follows the project's settings.</p>` : ""}
      <h4>The profile's file</h4>
      <p class="muted">Everything the profile sets, the form's fields included. What the form changed is put into this text when you save.</p>
      <textarea id="profile-text" spellcheck="false">${esc(e.text)}</textarea>
      <div class="row" style="margin-top:.6rem"><button type="button" data-act="save-profile">Save profile</button>
        <button class="quiet" type="button" data-act="close-profile">Close</button>
        ${e.own && !e.isNew ? `<button class="quiet danger" type="button" data-act="delete-profile">${e.builtin ? "Back to the built-in" : "Delete profile"}</button>` : ""}
        <span class="error" id="profile-error"></span></div></div>` : ""}

    <details class="whole" ${S.rawOpen ? "open" : ""}><summary>The whole file</summary>
      <p class="muted">rehearsal.toml as it is, for what the form does not show: per-song sections, a stem's chain. It is checked before it is saved, and the previous version kept as rehearsal.toml.bak.</p>
      <textarea id="raw-toml" spellcheck="false">${esc(s.toml)}</textarea>
      <div class="row" style="margin-top:.6rem"><button type="button" data-act="save-raw">Save file</button><span class="error" id="raw-error"></span></div>
    </details>`;
}

// -- a new project

function renderNew() {
  const d = S.draft, b = S.browse, pv = S.preview;
  $("#view").innerHTML = `<h1>New project</h1>
    <div class="stack" style="margin-top:1.6rem">
      <label><span>Name</span><input type="text" id="new-name" value="${esc(d.name)}" placeholder="Rehearsal, 5 November"></label>
      <div><label for="new-folder"><span class="muted" style="font-size:.87rem">Folder with the recordings</span></label>
        <div class="row" style="flex-wrap:nowrap;margin-top:.2rem"><input type="text" id="new-folder" value="${esc(d.folder)}" placeholder="${esc(S.home || "/home/you")}/rehearsals/2026-11-05">
          <button class="quiet" type="button" data-act="browse" style="white-space:nowrap">${S.browsing ? "Hide folders" : "Browse"}</button></div></div>
      ${S.browsing && b ? `<div class="browser">${b.parent ? `<button type="button" class="up" data-act="browse-to" data-path="${esc(b.parent)}">Up to ${esc(b.parent)}</button>` : ""}
        ${b.dirs.map((x) => `<button type="button" data-act="browse-to" data-path="${esc(x.path)}">${esc(x.name)}</button>`).join("") || `<span class="muted" style="padding:.4rem .8rem;display:block">No folders inside.</span>`}</div>` : ""}
      <div id="new-preview">${pv ? (pv.error ? `<p class="error">${esc(pv.error)}</p>` : `
        <p>${pv.takes.length ? `${plural(pv.takes.length, "take")} found.` : "No recordings found in this folder."}
          ${pv.has_settings ? "It already has a rehearsal.toml: the project uses it as it is." : ""}</p>
        <ul class="found">${pv.takes.map((t) => `<li><span class="take-title">${esc(t.title)}</span><span class="muted num">${esc(t.date)} ${clock(t.seconds)}</span>
          <span class="tracks">${t.tracks.map((x) => `<span class="track">${esc(x)}</span>`).join("")}</span></li>`).join("")}</ul>`) : ""}</div>
      ${pv && !pv.error && !pv.has_settings ? `<div><div class="muted" style="font-size:.87rem;margin-bottom:.3rem">Master every take with</div>
        <div class="chips">${pv.profiles.map((x) => `<label class="chip"><input type="checkbox" data-act="tick-draft" data-profile="${esc(x)}" ${d.profiles.has(x) ? "checked" : ""}>${esc(x)}</label>`).join("")}</div></div>` : ""}
      <div class="row"><button type="button" data-act="create-project" ${pv && !pv.error ? "" : "disabled"}>Create project</button>
        <a class="button quiet" href="#/">Cancel</a><span class="error" id="new-error"></span></div>
    </div>`;
}

async function previewFolder(folder) {
  S.draft.folder = folder;
  S.preview = null;
  if (folder.trim()) {
    try { S.preview = await api("GET", `/api/preview?folder=${encodeURIComponent(folder)}`); }
    catch (e) { S.preview = { error: e.message }; }
    // the name follows the folder until the person types one
    if (S.preview.folder && !S.draft.named) S.draft.name = S.preview.folder.split("/").filter(Boolean).pop() || "";
  }
  if (S.route.name === "new") renderNew();
}

async function browseTo(path) {
  try {
    S.browse = await api("GET", `/api/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`);
    S.browsing = true;
    await previewFolder(S.browse.path);
  } catch (e) { fail(e); }
}

// ---- actions ---------------------------------------------------------------------------------

async function startJob(separateOnly) {
  const p = S.project, sel = S.sel[p.id];
  try {
    const order = p.takes.map((t) => t.name).filter((n) => sel.takes.has(n));
    const profiles = p.profiles.map((x) => x.name).filter((n) => sel.profiles.has(n));
    await api("POST", "/api/jobs", { project: p.id, takes: order, profiles, ab: sel.ab, again: sel.again, separate_only: separateOnly });
    S.dismissed = 0;
    window.scrollTo({ top: 0, behavior: "smooth" });
    await poll();
    updateActions();
  } catch (e) { fail(e); }
}

function knobInto(edits, el) {
  const key = el.dataset.key;
  if (el.type === "checkbox") edits[key] = el.checked;
  else if (el.type === "number") { if (el.value !== "" && !Number.isNaN(Number(el.value))) edits[key] = Number(el.value); }
  else edits[key] = el.value;
}

const actions = {
  "tick-take": (el) => { const s = S.sel[S.project.id].takes; el.checked ? s.add(el.dataset.take) : s.delete(el.dataset.take); renderProject(); },
  "tick-all": (el) => { S.sel[S.project.id].takes = new Set(el.checked ? S.project.takes.map((t) => t.name) : []); renderProject(); },
  "tick-profile": (el) => { const s = S.sel[S.project.id].profiles; el.checked ? s.add(el.dataset.profile) : s.delete(el.dataset.profile); renderProject(); },
  "tick-ab": (el) => { S.sel[S.project.id].ab = el.checked; },
  "tick-again": (el) => { S.sel[S.project.id].again = el.checked; },
  remaster: () => startJob(false),
  separate: () => startJob(true),
  "cancel-job": async () => {
    if (!confirm("Stop the running job? Takes already finished keep their masters.")) return;
    try { await api("POST", "/api/job/cancel", {}); await poll(); updateActions(); } catch (e) { fail(e); }
  },
  "dismiss-job": () => { S.dismissed = S.job.started; sessionStorage.setItem("dismissed", String(S.dismissed)); renderJob(); },
  play: (el) => play({ src: el.dataset.src, label: el.dataset.label, take: el.dataset.take || "", offset: Number(el.dataset.offset || 0),
    sync: !!el.dataset.take, gainDb: 0 }),
  "play-node": (el) => { const n = nodeById(el.dataset.node); if (n) { selectNode(n); playNode(n); } },
  "graph-box": (el) => { const b = boxById(el.dataset.box); if (b) chooseBox(b, true); },
  view: (el) => {
    if (S.view === "blind" && el.dataset.view !== "blind") blindEnd();
    S.view = el.dataset.view;
    try { if (S.view !== "blind") localStorage.setItem("view", S.view); } catch (e) { /* private window */ }
    renderTake();
  },
  "blind-pick": (el) => { blindEnd(); const pair = [...document.querySelectorAll('[data-act="blind-pick"]')].map((x) => x.value);
    S.blind = { key: takeKey(), pair, order: null, on: 0, verdict: null, note: "", loading: false, error: "" }; renderTake(); },
  "blind-start": () => blindStart(),
  "blind-side": (el) => blindSide(Number(el.dataset.side)),
  "blind-toggle": () => { if (B.playing) blindPause(); else blindPlay(); },
  "blind-loop": (el) => { B.loop[Number(el.dataset.end)] = Math.round(blindPos() * 10) / 10;
    if (B.loop[0] != null && B.loop[1] != null && B.loop[1] - B.loop[0] < 0.3) B.loop[1 - Number(el.dataset.end)] = null;  // the other end was on the wrong side
    if (B.playing) blindPlay(blindPos()); else blindMark(); },
  "blind-loop-off": () => { const at = blindPos(); B.loop = [null, null]; if (B.playing) blindPlay(at); else blindMark(); },
  "blind-choose": async (el) => {
    const b = S.blind, choice = el.dataset.choice === "same" ? "same" : b.order[Number(el.dataset.choice)];
    try {
      S.project.listening = await api("POST", `/api/projects/${S.project.id}/verdicts`, { take: S.map.take.name, a: b.order[0], b: b.order[1],
        choice, note: $("#blind-note").value, loop_s: blindLoop() });
      b.verdict = choice;
      b.note = "";
      renderTake();
    } catch (e) { fail(e); }
  },
  "blind-again": () => { const b = S.blind; blindPause(); b.order = Math.random() < 0.5 ? [...b.pair] : [b.pair[1], b.pair[0]]; b.verdict = null; b.on = 0; renderTake(); },
  "blind-take-back": async (el) => {
    try { S.project.listening = await api("DELETE", `/api/projects/${S.project.id}/verdicts/${encodeURIComponent(el.dataset.id)}`);
      if (S.route.name === "take") renderTake(); else renderProject(); } catch (e) { fail(e); }
  },
  "tick-lined": (el) => {
    S.lined = el.checked;
    const n = S.map && [...S.map.nodes, ...S.map.extras].find((x) => nodeSrc(x) === P.src);
    if (n && !$("#player").hidden) setGain(S.lined ? S.map.stage_gain_db - n.level_db : 0);
  },
  "remaster-one": async () => {
    const m = S.map;
    try {
      await api("POST", "/api/jobs", { project: S.project.id, takes: [m.take.name], profiles: [m.profile],
        ab: m.nodes.some((n) => n.id === "master/ab"), again: true });  // asked for by name: made again whatever
      S.dismissed = 0;
      window.scrollTo({ top: 0, behavior: "smooth" });
      await poll();
      updateActions();
    } catch (e) { fail(e); }
  },
  "remove-project": async () => {
    if (!confirm(`Take "${S.project.name}" off the list? Its recordings, settings and masters stay where they are.`)) return;
    try { await api("DELETE", `/api/projects/${S.project.id}`); await poll(); location.hash = "#/"; } catch (e) { fail(e); }
  },

  // settings
  knob: (el) => knobInto(S.edits, el),
  pknob: (el) => knobInto(S.profileEdit.edits, el),
  "reset-knob": (el) => { S.edits[el.dataset.key] = null; renderSettings(); },
  "reset-pknob": (el) => { S.profileEdit.edits[el.dataset.key] = null; renderSettings(); },
  "tick-run": (el) => { el.checked ? S.run.add(el.dataset.profile) : S.run.delete(el.dataset.profile); },
  "save-settings": async () => {
    const run = S.project.profiles.map((x) => x.name).filter((n) => S.run.has(n));
    try {
      S.settings = await api("POST", `/api/projects/${S.project.id}/settings`, { values: S.edits, run });
      S.edits = {};
      S.run = new Set(S.settings.run);
      delete S.sel[S.project.id];
      await loadProject(true);
      renderSettings();
      toast("Saved to rehearsal.toml");
    } catch (e) { $("#settings-error").textContent = e.message; }
  },
  "save-raw": async () => {
    try {
      S.settings = await api("POST", `/api/projects/${S.project.id}/settings`, { toml: $("#raw-toml").value });
      S.edits = {};
      S.run = new Set(S.settings.run);
      delete S.sel[S.project.id];
      await loadProject(true);
      renderSettings();
      toast("Saved to rehearsal.toml");
    } catch (e) { $("#raw-error").textContent = e.message; }
  },
  "open-profile": async (el) => {
    try {
      S.profileEdit = { ...await api("GET", `/api/projects/${S.project.id}/profiles/${encodeURIComponent(el.dataset.profile)}`), edits: {} };
      renderSettings();
      $("#profile-editor").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) { fail(e); }
  },
  "new-profile": async () => {
    const name = $("#new-profile").value.trim(), from = $("#new-profile-from").value;
    if (!name) { $("#new-profile").focus(); return; }
    try {
      const p = await api("GET", `/api/projects/${S.project.id}/profiles/${encodeURIComponent(name)}?new=1${from ? `&copy=${encodeURIComponent(from)}` : ""}`);
      S.profileEdit = { ...p, isNew: true, edits: {} };
      renderSettings();
      $("#profile-editor").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) { fail(e); }
  },
  "save-profile": async () => {
    const e = S.profileEdit;
    try {
      const saved = await api("POST", `/api/projects/${S.project.id}/profiles/${encodeURIComponent(e.name)}`,
        { text: $("#profile-text").value, values: e.edits });
      S.profileEdit = { ...saved, edits: {} };
      await loadProject(true);
      renderSettings();
      toast(`Saved presets/${e.name}.toml`);
    } catch (err) { $("#profile-error").textContent = err.message; }
  },
  "delete-profile": async () => {
    const e = S.profileEdit;
    if (!confirm(e.builtin ? `Go back to the built-in "${e.name}"? This project's version of it is deleted. Masters already made with it stay.`
      : `Delete the profile "${e.name}"? Masters already made with it stay.`)) return;
    try {
      await api("DELETE", `/api/projects/${S.project.id}/profiles/${encodeURIComponent(e.name)}`);
      S.profileEdit = null;
      await loadProject(true);
      renderSettings();
    } catch (err) { $("#profile-error").textContent = err.message; }
  },
  "close-profile": () => { S.profileEdit = null; renderSettings(); },

  // new project
  browse: () => { if (S.browsing) { S.browsing = false; renderNew(); } else browseTo($("#new-folder").value.trim() || null); },
  "browse-to": (el) => browseTo(el.dataset.path),
  "tick-draft": (el) => { el.checked ? S.draft.profiles.add(el.dataset.profile) : S.draft.profiles.delete(el.dataset.profile); },
  "create-project": async () => {
    try {
      const p = await api("POST", "/api/projects", { name: $("#new-name").value, folder: S.draft.folder,
                                                    profiles: (S.preview.profiles || []).filter((x) => S.draft.profiles.has(x)) });
      S.draft = { name: "", folder: "", profiles: new Set(["default"]) };
      S.preview = S.browse = null;
      S.browsing = false;
      await poll();
      location.hash = `#/p/${p.id}`;
    } catch (e) { $("#new-error").textContent = e.message; }
  },
};

document.addEventListener("click", (ev) => {
  const jump = ev.target.closest("[data-jump]");  // the guide's contents: the app's own addresses are the views
  if (jump) { ev.preventDefault(); document.getElementById(jump.dataset.jump)?.scrollIntoView({ behavior: "smooth" }); return; }
  const el = ev.target.closest("[data-act]");
  if (!el || el.matches("input, select, textarea")) return;
  actions[el.dataset.act]?.(el);
});
document.addEventListener("change", (ev) => {
  const el = ev.target;
  if (el.id === "new-folder") return previewFolder(el.value.trim());
  if (el.id === "new-name") return;
  if (el.matches("[data-act]")) actions[el.dataset.act]?.(el);
});
document.addEventListener("input", (ev) => {
  if (ev.target.matches('input[type="number"], input[type="text"]') && /^p?knob$/.test(ev.target.dataset.act || "")) actions[ev.target.dataset.act](ev.target);
  if (ev.target.id === "new-name") { S.draft.name = ev.target.value; S.draft.named = ev.target.value.trim() !== ""; }
  if (ev.target.id === "blind-seek") blindSeek(Number(ev.target.value));
  if (ev.target.id === "blind-note" && S.blind) S.blind.note = ev.target.value;
});
document.addEventListener("keydown", (ev) => {  // comparing blind: the sides, play and pause, the loop
  const b = S.blind;
  if (S.route.name !== "take" || S.view !== "blind" || !b || !b.order || ev.altKey || ev.ctrlKey || ev.metaKey) return;
  if (ev.target.matches && ev.target.matches("input[type=text], textarea, select")) return;
  const k = ev.key.toLowerCase();
  // Space plays and pauses, also right after A or B was pressed with the mouse; on any other button it presses that button
  if (k === " " && ev.target.matches && ev.target.matches("button, a, summary") && !ev.target.matches(".side")) return;
  const act = { a: () => blindSide(0), b: () => blindSide(1), x: () => blindSide(1 - b.on), " ": () => actions["blind-toggle"](),
    "[": () => actions["blind-loop"]({ dataset: { end: "0" } }), "]": () => actions["blind-loop"]({ dataset: { end: "1" } }) }[k];
  if (!act) return;
  ev.preventDefault();
  act();
});
document.addEventListener("keyup", (ev) => {  // Space on a side played or paused (above): it must not press the side as well
  if (ev.key === " " && S.view === "blind" && ev.target.matches && ev.target.matches(".side")) ev.preventDefault();
});
document.addEventListener("toggle", (ev) => {
  if (ev.target.matches("details.log")) { S.logOpen = ev.target.open; if (ev.target.open) $("#job-log").scrollTop = 1e9; }
  if (ev.target.matches("details.whole")) S.rawOpen = ev.target.open;
  if (ev.target.matches("details.extras")) S.extrasOpen = ev.target.open;
}, true);
document.addEventListener("keydown", (ev) => {  // on a box of the graph: Enter and Space press it, the arrows go to its neighbours
  if (!ev.target.matches || !ev.target.matches(".gbox") || ev.altKey || ev.ctrlKey || ev.metaKey) return;
  const here = boxById(ev.target.dataset.box), a = $("#audio");
  const move = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[ev.key];
  if (!here || !(move || ev.key === "Enter" || ev.key === " ")) return;
  ev.preventDefault();
  if (!move) return chooseBox(here, true);
  const next = nextBox(here, ...move);
  if (!next) return;
  [...document.querySelectorAll(".gbox")].find((el) => el.dataset.box === next.id)?.focus();
  chooseBox(next, !a.paused && !a.ended && !$("#player").hidden);  // listening: the next box takes over
});
document.addEventListener("keydown", (ev) => {  // on a pad of the take's page: the arrows go to its neighbours
  const move = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[ev.key];
  if (!move || !ev.target.matches || !ev.target.matches(".map .pad") || ev.altKey || ev.ctrlKey || ev.metaKey) return;
  ev.preventDefault();
  const here = nodeById(ev.target.dataset.node);
  if (here && here.id !== S.cur) selectNode(here);
  const n = nextPad(...move), a = $("#audio");
  if (!n) return;
  selectNode(n);
  [...document.querySelectorAll(".map .pad")].find((b) => b.dataset.node === n.id)?.focus();
  if (!a.paused && !a.ended && !$("#player").hidden) playNode(n);  // listening: the next pad takes over
});
const audio = $("#audio");
audio.addEventListener("timeupdate", () => { if (P.sync && !P.settling && !audio.seeking) P.pos = Math.max(0, audio.currentTime - P.offset); });
audio.addEventListener("playing", () => { P.settling = false; markPlaying(); });
audio.addEventListener("pause", markPlaying);
audio.addEventListener("ended", () => { P.pos = 0; markPlaying(); });
audio.addEventListener("error", () => { if (P.src && !$("#player").hidden) toast("This file does not play in this browser.", true); });
$("#player-close").addEventListener("click", () => {
  audio.pause();
  $("#player").hidden = true;
  document.body.classList.remove("playing");
  markPlaying();
});
window.addEventListener("hashchange", navigate);

try { if (localStorage.getItem("view") === "graph") S.view = "graph"; } catch (e) { /* private window */ }
poll().then(navigate);
