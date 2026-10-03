"use strict";
/* =====================================================================
   compute-odds.js  (Netlify Function, plain JavaScript, no dependencies)

   1. Pulls prematch tennis fixtures from Pinnwire (key = env var PINNWIRE_KEY)
   2. Keeps ATP 1000, WTA 1000 and ATP Challenger singles only
   3. Reads the market's game spread + total-games ladders and reverse-engineers
      the serve strengths that reproduce them (exact point -> game -> set -> match model)
   4. Returns win probabilities, set-handicap prices and a pick for each match.
      If Pinnacle has no moneyline, odds are estimated from the model.
   ===================================================================== */

/* ---------------------------- CONFIG ---------------------------- */
const CFG = {
  URL: "https://pinnwire.com/kit/v1/prematch/fixtures?sport_id=2",
  CACHE_MS: 20 * 60 * 1000,      // free plan = 100 requests/day, so refresh at most every 20 min
  MARGIN: 0.045,                 // overround used when we have to estimate odds
  EV_EDGE: 0.03,                 // "VALUE" needs model EV above 3% against a real moneyline
  PICKEM: [1.70, 2.50],          // both players inside this range = balanced pick'em
  MAX_MATCHES: 40,
  LOOKAHEAD_H: 72,
  /* Optional player stats blend. Pinnwire has no serve/return stats, so this stays
     off (weight 0) until you fill STATS below. */
  WEIGHT: { "ATP 1000": 0.15, "WTA 1000": 0.15, "Challenger": 0.75 },
  TOUR_SERVE: { "ATP 1000": 0.64, "WTA 1000": 0.56, "Challenger": 0.63 }, // average serve-point win %
  CAL_GAMMA: 1.0                 // 1 = off. Below 1 shrinks probabilities toward 50% (see notes)
};
/* "Player Name": { serve: 0.62, ret: 0.40 }  (serve-point win %, return-point win %) */
const STATS = {};

/* Which tournaments count as 1000s. Edit these if Pinnwire names a venue differently. */
const ATP_1000 = /(indian wells|miami|monte[- ]?carlo|madrid|rome|canada|toronto|montreal|cincinnati|shanghai|paris)/i;
const WTA_1000 = /(doha|dubai|indian wells|miami|madrid|rome|canada|toronto|montreal|cincinnati|beijing|wuhan)/i;
const ROUND_RE = /\s+-\s+(R\d+|Q\d*|QF|SF|F|Final|Qualifying)\b.*$/i;

/* ---------------------------- TENNIS MATH ---------------------------- */
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

/* exact chance the server holds a game, given point-win probability s */
function hold(s) {
  s = clamp(s, 0.01, 0.99);
  const q = 1 - s;
  return Math.pow(s, 4) * (1 + 4 * q + 10 * q * q) +
         20 * Math.pow(s, 3) * Math.pow(q, 3) * (s * s / (s * s + q * q));
}
function holdToPoint(g) {
  let lo = 0.01, hi = 0.99;
  g = clamp(g, 0.02, 0.995);
  for (let n = 0; n < 50; n++) { const m = (lo + hi) / 2; if (hold(m) < g) lo = m; else hi = m; }
  return (lo + hi) / 2;
}

/* tiebreak: chance A wins. aFirst = A serves the first point. Serve pattern 1,2,2,2,... */
function tiebreak(sA, sB, aFirst) {
  const dp = Array.from({ length: 7 }, () => Array(7).fill(0));
  dp[0][0] = 1;
  let win = 0;
  for (let n = 0; n < 12; n++) {
    for (let a = 0; a <= 6; a++) {
      const b = n - a;
      if (b < 0 || b > 6) continue;
      const p = dp[a][b];
      if (!p) continue;
      const aServes = ((((n + 1) >> 1) % 2) === 0) === aFirst;
      const pa = aServes ? sA : 1 - sB;
      if (a + 1 === 7) win += p * pa; else dp[a + 1][b] += p * pa;
      if (b + 1 !== 7) dp[a][b + 1] += p * (1 - pa);
    }
  }
  const x = sA * (1 - sB), y = (1 - sA) * sB;   // from 6-6, two-point rounds
  return win + dp[6][6] * x / (x + y);
}

/* distribution of final set scores [gamesA, gamesB, prob]; each game alternates server */
function setDist(sA, sB, aFirst) {
  const hA = hold(sA), hB = hold(sB), tb = tiebreak(sA, sB, aFirst);
  const D = Array.from({ length: 8 }, () => Array(8).fill(0));
  D[0][0] = 1;
  const out = [];
  for (let n = 0; n <= 12; n++) {
    for (let i = 0; i <= 7; i++) {
      const j = n - i;
      if (j < 0 || j > 7) continue;
      const p = D[i][j];
      if (!p) continue;
      if ((i === 6 && j <= 4) || (j === 6 && i <= 4) || (i === 7 && j === 5) || (j === 7 && i === 5)) { out.push([i, j, p]); continue; }
      if (i === 6 && j === 6) { out.push([7, 6, p * tb], [6, 7, p * (1 - tb)]); continue; }
      const aServes = (n % 2 === 0) === aFirst;
      const pa = aServes ? hA : 1 - hB;
      D[i + 1][j] += p * pa;
      D[i][j + 1] += p * (1 - pa);
    }
  }
  return out;
}

/* best-of-3 match: exact win chance, expected total games, expected game margin, set-score chances.
   The next set is served first by whoever did not serve last; who serves first overall is a coin toss. */
function matchStats(sA, sB, hur) {
  hur = hur || {};
  let pm = 0, pt = 0;
  const dist = { t: setDist(sA, sB, true), f: setDist(sA, sB, false) };
  const flip = (f, games) => (games % 2 === 0 ? f : !f);
  let pA = 0, eT = 0, eM = 0, A20 = 0, A21 = 0, B20 = 0, B21 = 0;
  for (const f1 of [true, false]) {
    for (const [a1, b1, p1] of dist[f1 ? "t" : "f"]) {
      const f2 = flip(f1, a1 + b1);
      for (const [a2, b2, p2] of dist[f2 ? "t" : "f"]) {
        const w = 0.5 * p1 * p2, s1 = a1 > b1, s2 = a2 > b2;
        if (s1 === s2) {
          const g = a1 + b1 + a2 + b2, m = a1 - b1 + a2 - b2;
          if (s1) { pA += w; A20 += w; } else B20 += w;
          eT += w * g; eM += w * m;
          if (m > hur.m) pm += w;
          if (g > hur.t) pt += w;
        } else {
          const f3 = flip(f2, a2 + b2);
          for (const [a3, b3, p3] of dist[f3 ? "t" : "f"]) {
            const q = w * p3, g = a1 + b1 + a2 + b2 + a3 + b3, m = a1 - b1 + a2 - b2 + a3 - b3;
            if (a3 > b3) { pA += q; A21 += q; } else B21 += q;
            eT += q * g; eM += q * m;
            if (m > hur.m) pm += q;
            if (g > hur.t) pt += q;
          }
        }
      }
    }
  }
  return { pA, eT, eM, A20, A21, B20, B21, pm, pt };
}

/* find serve-point strengths (sA, sB) that reproduce the market's margin and total */
/* Fit to the market's actual line probabilities (not to averages): P(margin > -hdp) must equal the
   devigged chance the home side covers, and P(total games > line) the devigged chance of the over. */
function solveMarket(L) {
  const hur = { m: -L.hLine, t: L.tLine };
  const W = 0.5;   // the margin pins down win probability, so it counts double; i.i.d. sets cannot always match the total too
  const F = (x) => { const m = matchStats(x[0], x[1], hur); return [m.pm - L.cover, W * (m.pt - L.pOver)]; };
  const norm = (r) => Math.hypot(r[0], r[1]);
  const lo = 0.30, hi = 0.88;
  /* coarse grid first (the total is not monotone in strength, so a blind Newton start can stall) */
  let x = [0.64, 0.60], r = F(x);
  for (let a = 0.30; a <= 0.88; a += 0.04) for (let b = 0.30; b <= 0.88; b += 0.04) {
    const rr = F([a, b]);
    if (norm(rr) < norm(r)) { x = [a, b]; r = rr; }
  }
  /* then Levenberg-Marquardt polish */
  let lam = 1e-3;
  for (let it = 0; it < 60 && norm(r) > 1e-7; it++) {
    const h = 1e-4, rA = F([x[0] + h, x[1]]), rB = F([x[0], x[1] + h]);
    const J = [[(rA[0] - r[0]) / h, (rB[0] - r[0]) / h], [(rA[1] - r[1]) / h, (rB[1] - r[1]) / h]];
    const A = [[J[0][0] * J[0][0] + J[1][0] * J[1][0] + lam, J[0][0] * J[0][1] + J[1][0] * J[1][1]],
               [J[0][0] * J[0][1] + J[1][0] * J[1][1], J[0][1] * J[0][1] + J[1][1] * J[1][1] + lam]];
    const g = [J[0][0] * r[0] + J[1][0] * r[1], J[0][1] * r[0] + J[1][1] * r[1]];
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (!isFinite(det) || Math.abs(det) < 1e-12) break;
    const dx = [(A[1][1] * g[0] - A[0][1] * g[1]) / det, (-A[1][0] * g[0] + A[0][0] * g[1]) / det];
    const nx = [clamp(x[0] - dx[0], lo, hi), clamp(x[1] - dx[1], lo, hi)], nr = F(nx);
    if (norm(nr) < norm(r)) { x = nx; r = nr; lam = Math.max(lam / 3, 1e-9); } else { lam *= 4; if (lam > 1e6) break; }
  }
  return Math.abs(r[0]) < 0.02 ? { sA: x[0], sB: x[1], errM: r[0], errT: r[1] / W } : null;
}

/* optional blend with player stats (only when both players have stats) */
function blend(sA, sB, cat, a, b) {
  if (!a || !b || !CFG.WEIGHT[cat]) return { sA, sB, w: 0 };
  const w = CFG.WEIGHT[cat], ts = CFG.TOUR_SERVE[cat];
  const eA = clamp(a.serve - (b.ret - (1 - ts)), 0.3, 0.9);   // my serve vs their return, relative to tour average
  const eB = clamp(b.serve - (a.ret - (1 - ts)), 0.3, 0.9);
  return { sA: holdToPoint(w * hold(eA) + (1 - w) * hold(sA)), sB: holdToPoint(w * hold(eB) + (1 - w) * hold(sB)), w };
}

/* ---------------------------- READING THE FEED ---------------------------- */
const clean = (s) => String(s || "").replace(/\s*\(games\)\s*$/i, "").trim();
const devig2 = (x, y) => (1 / x) / (1 / x + 1 / y);

function categoryOf(lg) {
  const n = String(lg || "").replace(ROUND_RE, "");
  if (/doubles/i.test(n)) return "";
  if (/^ATP Challenger/i.test(n)) return "Challenger";
  if (/^ATP\s/i.test(n) && ATP_1000.test(n)) return "ATP 1000";
  if (/^WTA\s/i.test(n) && !/125/.test(n) && WTA_1000.test(n)) return "WTA 1000";
  return "";
}

/* longest run of points whose y strictly increases (drops stray lines such as set handicaps) */
function lis(pts) {
  const n = pts.length, len = Array(n).fill(1), prev = Array(n).fill(-1);
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) if (pts[j].y < pts[i].y && len[j] + 1 > len[i]) { len[i] = len[j] + 1; prev[i] = j; }
  let k = len.indexOf(Math.max.apply(null, len)); const out = [];
  while (k >= 0) { out.unshift(pts[k]); k = prev[k]; }
  return out;
}
/* x where the devigged chance crosses 50% */
function crossing(pts) {
  if (pts.length < 2) return null;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if (a.y <= 0.5 && b.y >= 0.5) return a.x + (0.5 - a.y) * (b.x - a.x) / (b.y - a.y);
  }
  const [a, b] = pts[0].y > 0.5 ? [pts[0], pts[1]] : [pts[pts.length - 2], pts[pts.length - 1]];
  return clamp(a.x + (0.5 - a.y) * (b.x - a.x) / (b.y - a.y), pts[0].x - 1.5, pts[pts.length - 1].x + 1.5);
}
const isHalf = (v) => Math.abs(Math.abs(v) % 1 - 0.5) < 1e-9;

/* Pinnwire mixes set handicaps (+-1.5) into spreads and a sets total (2.5) into totals,
   so: keep half-point lines, drop +-1.5 when other lines exist, then keep only a consistent ladder. */
function ladders(p0) {
  let sp = Object.values(p0.spreads || {}).filter((l) => l && +l.home > 1 && +l.away > 1)
    .map((l) => ({ x: +l.hdp, y: devig2(+l.home, +l.away) }));
  let tt = Object.values(p0.totals || {}).filter((l) => l && +l.points >= 10 && +l.over > 1 && +l.under > 1)
    .map((l) => ({ x: +l.points, y: 1 - devig2(+l.over, +l.under) }));
  const prep = (arr, dropSets) => {
    let a = arr.slice();
    const halves = a.filter((p) => isHalf(p.x));
    if (halves.length >= 2) a = halves;
    if (dropSets) { const rest = a.filter((p) => Math.abs(Math.abs(p.x) - 1.5) > 1e-9); if (rest.length >= 3) a = rest; }
    a.sort((p, q) => p.x - q.x);
    return lis(a);
  };
  const ps = prep(sp, true), pt = prep(tt, false);
  const s = crossing(ps), t = crossing(pt);
  if (s == null || t == null) return null;
  const near = (a) => a.reduce((b, p) => (Math.abs(p.y - 0.5) < Math.abs(b.y - 0.5) ? p : b));
  const ns = near(ps), nt = near(pt);
  return { hdp: s, total: t, hLine: ns.x, cover: ns.y, tLine: nt.x, pOver: 1 - nt.y };
}

/* ---------------------------- BUILD ---------------------------- */
function build(events, nowMs, generatedAt) {
  const skipped = {}, bump = (k) => { skipped[k] = (skipped[k] || 0) + 1; };

  /* real moneylines come from whichever event carries one; the "(Games)" copies usually don't */
  const mlIdx = {};
  events.forEach((ev) => {
    const ml = ev.periods && ev.periods.num_0 && ev.periods.num_0.money_line;
    if (!ml || !(+ml.home > 1) || !(+ml.away > 1)) return;
    const h = clean(ev.home), a = clean(ev.away);
    mlIdx[h + "|" + a] = [+ml.home, +ml.away];
    mlIdx[a + "|" + h] = [+ml.away, +ml.home];
  });

  const seen = {}, cands = [];
  events.forEach((ev) => {
    const cat = categoryOf(ev.league_name);
    if (!cat) { bump("not ATP/WTA 1000 or Challenger"); return; }
    if (/\//.test(String(ev.home) + String(ev.away))) { bump("doubles"); return; }
    const st = Date.parse(ev.starts || ev.start_ts);
    if (!(st > nowMs) || st > nowMs + CFG.LOOKAHEAD_H * 3600e3) { bump("not starting soon"); return; }
    const p0 = ev.periods && ev.periods.num_0;
    if (!p0) { bump("no full-match market"); return; }
    const home = clean(ev.home), away = clean(ev.away), key = home + "|" + away;
    if (seen[key] || seen[away + "|" + home]) return;
    const lad = ladders(p0);
    if (!lad) { bump("not enough spread/total lines"); return; }
    seen[key] = 1;
    cands.push({ ev, cat, st, home, away, lad });
  });
  cands.sort((a, b) => a.st - b.st);

  const matches = [];
  cands.slice(0, CFG.MAX_MATCHES).forEach((c) => {
    const margin = -c.lad.hdp, total = c.lad.total;           // home favourite = negative hdp
    const sol = solveMarket(c.lad);
    if (!sol) { bump("model did not fit the lines"); return; }
    const bl = blend(sol.sA, sol.sB, c.cat, STATS[c.home], STATS[c.away]);
    const m = matchStats(bl.sA, bl.sB);
    const pA = clamp(0.5 + (m.pA - 0.5) * CFG.CAL_GAMMA, 0.02, 0.98), pB = 1 - pA;

    const real = mlIdx[c.home + "|" + c.away] || null;
    const odds = real ? real.slice() : [1 / (pA * (1 + CFG.MARGIN)), 1 / (pB * (1 + CFG.MARGIN))];
    const o = odds.map((x) => Math.round(x * 100) / 100);
    const ev = [pA * o[0] - 1, pB * o[1] - 1];
    const mlProb = real ? devig2(real[0], real[1]) : null;
    const pickem = o.every((x) => x >= CFG.PICKEM[0] && x <= CFG.PICKEM[1]);

    let side = pA >= pB ? 0 : 1, status = real ? "LEAN" : "MODEL";
    if (real) { const b = ev[0] >= ev[1] ? 0 : 1; if (ev[b] > CFG.EV_EDGE) { side = b; status = "VALUE"; } }
    const name = side ? c.away : c.home, pp = side ? pB : pA;
    const fav = pA >= pB ? c.home : c.away;
    let reason = "Lines imply " + c.home + " " + (margin >= 0 ? "-" : "+") + Math.abs(margin).toFixed(1) +
      " games over " + total.toFixed(1) + " total. Model: " + c.home + " " + (pA * 100).toFixed(1) + "%, " + fav + " favoured.";
    reason += real ? " Pinnacle moneyline implies " + (mlProb * 100).toFixed(1) + "% for " + c.home + "; model EV on the pick " + (ev[side] >= 0 ? "+" : "") + (ev[side] * 100).toFixed(1) + "%."
                   : " No moneyline posted, so odds are estimated from the model with a " + (CFG.MARGIN * 100).toFixed(1) + "% margin.";
    if (Math.abs(sol.errT) > 0.03) reason += " Total-games line fitted loosely (off by " + (Math.abs(sol.errT) * 100).toFixed(0) + " pts).";
    if (bl.w) reason += " Player stats blended at " + Math.round(bl.w * 100) + "%.";

    const lg = String(c.ev.league_name), rd = (lg.match(ROUND_RE) || [])[1] || "";
    const fair = [1 / pA, 1 / pB], book = [1 / (pA * (1 + CFG.MARGIN)), 1 / (pB * (1 + CFG.MARGIN))];
    const r2 = (x) => Math.round(x * 100) / 100;
    const model = {
      fair: fair.map(r2), book: book.map(r2),
      total: Math.round(m.eT * 10) / 10, spread: Math.round(m.eM * 10) / 10,          // spread = expected games home minus away
      holds: [Math.round(hold(bl.sA) * 1000) / 10, Math.round(hold(bl.sB) * 1000) / 10],
      mktTotal: Math.round(total * 10) / 10, mktSpread: Math.round(margin * 10) / 10,
      loose: Math.abs(sol.errT) > 0.03
    };
    const pick = {
      model, odds: o, prediction: status, market: name + " Match Winner", pickOdds: o[side], prob: Math.round(pp * 1000) / 10,
      reason, cat: c.cat, pickem, src: real ? "Pinnacle moneyline" : "estimated from model",
      sets: { A20: m.A20, A21: m.A21, B20: m.B20, B21: m.B21 }
    };
    matches.push([c.home, c.away, new Date(c.st).toISOString(), [Math.round(pA * 1000) / 10, Math.round(pB * 1000) / 10],
      lg.replace(ROUND_RE, "") + (rd ? " " + rd : "") + " · " + c.cat, pick, "e" + c.ev.event_id]);
  });

  return { matches, meta: { generated_at: generatedAt || new Date(nowMs).toISOString(), scanned: events.length, kept: matches.length, skipped } };
}

/* ---------------------------- HANDLER ---------------------------- */
let CACHE = null;
const reply = (status, body, ok) => ({
  statusCode: status,
  headers: Object.assign({ "Content-Type": "application/json" }, ok
    ? { "Cache-Control": "public, max-age=300", "Netlify-CDN-Cache-Control": "public, s-maxage=1200, stale-while-revalidate=3600" }
    : { "Cache-Control": "no-store" }),
  body: JSON.stringify(body)
});

exports.handler = async function () {
  const key = process.env.PINNWIRE_KEY;
  if (!key) return reply(500, { error: "PINNWIRE_KEY is not set in the Netlify environment variables" });
  if (CACHE && Date.now() - CACHE.ts < CFG.CACHE_MS) return reply(200, CACHE.payload, true);
  try {
    const res = await fetch(CFG.URL, { headers: { "x-api-key": key, "User-Agent": "CallIt/1.0" }, signal: AbortSignal.timeout(8000) });
    if (res.status === 401) return reply(502, { error: "Pinnwire rejected the key (401). Check PINNWIRE_KEY." });
    if (res.status === 429) throw new Error("Pinnwire rate limit hit (retry in " + (res.headers.get("retry-after") || "a while") + "s)");
    if (!res.ok) throw new Error("Pinnwire HTTP " + res.status);
    const data = await res.json();
    const payload = build(Array.isArray(data.events) ? data.events : [], Date.now(), data.generated_at);
    CACHE = { ts: Date.now(), payload };
    return reply(200, payload, true);
  } catch (e) {
    if (CACHE) return reply(200, Object.assign({}, CACHE.payload, { stale: true, warning: String(e.message || e) }), false);
    return reply(502, { error: String(e.message || e) });
  }
};

exports._test = { hold, holdToPoint, tiebreak, setDist, matchStats, solveMarket, ladders, build, categoryOf };
