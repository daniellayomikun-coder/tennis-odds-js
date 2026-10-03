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
  CAL_GAMMA: 1.0                 // 1 = off. Below 1 shrinks probabilities toward 50% (not recommended)
};
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
  const gd = Array(60).fill(0), md = Array(81).fill(0);   // distributions of total games and of game margin (offset 40)
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
          eT += w * g; eM += w * m; gd[g] += w; md[m + 40] += w;
          if (m > hur.m) pm += w;
          if (g > hur.t) pt += w;
        } else {
          const f3 = flip(f2, a2 + b2);
          for (const [a3, b3, p3] of dist[f3 ? "t" : "f"]) {
            const q = w * p3, g = a1 + b1 + a2 + b2 + a3 + b3, m = a1 - b1 + a2 - b2 + a3 - b3;
            if (a3 > b3) { pA += q; A21 += q; } else B21 += q;
            eT += q * g; eM += q * m; gd[g] += q; md[m + 40] += q;
            if (m > hur.m) pm += q;
            if (g > hur.t) pt += q;
          }
        }
      }
    }
  }
  return { pA, eT, eM, A20, A21, B20, B21, pm, pt, gd, md };
}

/* Matches are not i.i.d. coin-flip sets: a player who is "on" in set 1 is usually on in set 2. We model that with a
   match-level form shock d ~ N(0, tau^2): player A serves at sA + d and B at sB - d. More tau = more straight-set
   results and wider game margins, which is what real totals and handicap lines show. 5-point Gauss-Hermite average. */
const GH = [[-2.857, 0.01126], [-1.3556, 0.2221], [0, 0.5333], [1.3556, 0.2221], [2.857, 0.01126]];
function mix(sA, sB, tau, hur) {
  const nodes = tau > 0 ? GH : [[0, 1]], ws = nodes.reduce((a, n) => a + n[1], 0);
  const out = { pA: 0, eT: 0, eM: 0, A20: 0, A21: 0, B20: 0, B21: 0, pm: 0, pt: 0, gd: Array(60).fill(0), md: Array(81).fill(0) };
  for (const [z, w] of nodes) {
    const ww = w / ws, d = tau * z, r = matchStats(clamp(sA + d, 0.2, 0.92), clamp(sB - d, 0.2, 0.92), hur);
    for (const k of ["pA", "eT", "eM", "A20", "A21", "B20", "B21", "pm", "pt"]) out[k] += ww * r[k];
    for (let i = 0; i < 60; i++) out.gd[i] += ww * r.gd[i];
    for (let i = 0; i < 81; i++) out.md[i] += ww * r.md[i];
  }
  return out;
}

/* the total-games value where P(over) = 50%, on the same half-point scale as the bookmaker lines */
function medianTotal(gd) {
  let cum = 0;
  for (let k = 0; k < gd.length - 1; k++) {
    const before = 1 - cum;          // P(total > k - 0.5)
    cum += gd[k];
    const after = 1 - cum;           // P(total > k + 0.5)
    if (before >= 0.5 && after <= 0.5) return (k - 0.5) + (before - 0.5) / Math.max(before - after, 1e-9);
  }
  return null;
}

/* find serve-point strengths (sA, sB) that reproduce the market's margin and total */
/* the game margin (home minus away) where P(margin beats it) = 50%, on the same scale as the bookmaker's handicap lines */
function medianMargin(md) {
  let prev = null;
  for (let k = -39.5; k <= 39.5; k += 1) {
    let above = 0;
    for (let m = Math.ceil(k); m <= 40; m++) above += md[m + 40];
    if (prev && prev.p >= 0.5 && above <= 0.5) return prev.k + (prev.p - 0.5) / Math.max(prev.p - above, 1e-9);
    prev = { k, p: above };
  }
  return null;
}

/* Fit to the market's actual line probabilities with three knobs:
   d   = how much better A serves than B   -> pinned by P(margin beats the main handicap line) = market chance (the win probability)
   tau = match-level form variance          -> pinned by P(total games beats the main line)    = market chance
   S   = overall serve level                -> starts at the tour average, only moved if tau alone cannot reach the total line
   Everything stays in realistic ranges, so a low total can no longer be "explained" by a 99% hold. */
const TOUR_S = { "ATP 1000": 0.64, "WTA 1000": 0.565, "Challenger": 0.62 };
function solveMarket(L, S0) {
  const hur = { m: -L.hLine, t: L.tLine };
  const solveD = (S, tau) => {
    let lo = -0.34, hi = 0.34;
    const f = (d) => mix(S + d / 2, S - d / 2, tau, hur).pm;
    if (f(lo) > L.cover || f(hi) < L.cover) return null;
    for (let n = 0; n < 11; n++) { const m = (lo + hi) / 2; if (f(m) < L.cover) lo = m; else hi = m; }
    return (lo + hi) / 2;
  };
  const at = (S, tau) => {
    const d = solveD(S, tau);
    if (d == null) return null;
    const r = mix(S + d / 2, S - d / 2, tau, hur);
    return { S, tau, d, sA: S + d / 2, sB: S - d / 2, errM: r.pm - L.cover, errT: r.pt - L.pOver };
  };
  let best = null;
  for (const S of [S0, S0 - 0.03, S0 + 0.03, S0 - 0.06, S0 + 0.06, S0 - 0.09, S0 + 0.09, S0 - 0.12, S0 + 0.12]) {
    if (S < 0.45 || S > 0.76) continue;
    const t0 = at(S, 0), t1 = at(S, 0.10);
    if (!t0 || !t1) continue;
    let c;
    if (t0.errT <= 0) c = t0;                     // even without form variance there are too few games: needs a higher S
    else if (t1.errT >= 0) c = t1;                // even with maximum variance there are too many games: needs a lower S
    else {
      let lo = 0, hi = 0.10, mid = null;
      for (let n = 0; n < 9; n++) { const m = (lo + hi) / 2, x = at(S, m); if (!x) break; mid = x; if (x.errT > 0) lo = m; else hi = m; }
      c = mid || t1;
    }
    if (!best || Math.abs(c.errT) < Math.abs(best.errT)) best = c;
    if (Math.abs(best.errT) < 0.012) break;       // close enough: keep the S nearest the tour average
  }
  return best && Math.abs(best.errM) < 0.01 ? best : null;
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
    .map((l) => ({ x: +l.hdp, y: devig2(+l.home, +l.away), a: +l.home, b: +l.away }));
  let tt = Object.values(p0.totals || {}).filter((l) => l && +l.points >= 10 && +l.over > 1 && +l.under > 1)
    .map((l) => ({ x: +l.points, y: 1 - devig2(+l.over, +l.under), a: +l.over, b: +l.under }));
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
  return { hdp: s, total: t, hLine: ns.x, cover: ns.y, tLine: nt.x, pOver: 1 - nt.y, hPts: ps, tPts: pt };
}

/* ---------------------------- BUILD ---------------------------- */
const R1 = (x) => Math.round(x * 10) / 10, R2 = (x) => Math.round(x * 100) / 100;
const SOLVED = new Map();                      // fitted models survive between warm invocations

/* every market we can price from the fitted match distribution */
function marketsFor(r, home, away, fav, ladd, mlOdds) {
  const out = [], pA = r.pA, pB = 1 - pA, H = home, A = away;
  const row = (g, l, p, o, s, src, all) => {
    if (!(p > 0.0005 && p < 0.9995)) return;
    out.push({ k: g.replace(/\W/g, "").slice(0, 6) + out.length, g, l, p: R1(p * 100), o: Math.max(1.01, R2(o || 1 / p)), s, r: src || "f", a: all ? 1 : 0 });
  };
  const fh = fav === 0;                                        // is home the favourite?
  row("Winner", H, pA, mlOdds ? mlOdds[0] : 0, fh ? "f" : "u", mlOdds ? "p" : "f");
  row("Winner", A, pB, mlOdds ? mlOdds[1] : 0, fh ? "u" : "f", mlOdds ? "p" : "f");
  /* set handicap */
  row("Set handicap", H + " -1.5 sets", r.A20, 0, fh ? "f" : "u");
  row("Set handicap", A + " -1.5 sets", r.B20, 0, fh ? "u" : "f");
  row("Set handicap", H + " +1.5 sets", 1 - r.B20, 0, fh ? "f" : "u");
  row("Set handicap", A + " +1.5 sets", 1 - r.A20, 0, fh ? "u" : "f");
  /* total games (real Pinnacle price when that exact line is on the feed) */
  const tp = {}; (ladd.tPts || []).forEach((q) => { tp[q.x] = q; });
  const med = Math.round(r.medTotal || 22);
  for (let k = med - 7; k <= med + 6; k++) {
    const L = k + 0.5; let over = 0;
    for (let g = 0; g < r.gd.length; g++) if (g > L) over += r.gd[g];
    const q = tp[L];
    if (over >= 0.12 && over <= 0.9) row("Total games", "Over " + L, over, q ? q.a : 0, "n", q ? "p" : "f");
    if (1 - over >= 0.12 && 1 - over <= 0.9) row("Total games", "Under " + L, 1 - over, q ? q.b : 0, "n", q ? "p" : "f");
  }
  /* game handicap (spread): home covers hdp when margin + hdp > 0 */
  const hp = {}; (ladd.hPts || []).forEach((q) => { hp[q.x] = q; });
  for (let k = -9.5; k <= 9.5; k += 1) {
    let cov = 0; for (let m = -40; m <= 40; m++) if (m + k > 0) cov += r.md[m + 40];
    const q = hp[k], sg = (v) => (v > 0 ? "+" : "") + v;
    if (cov >= 0.12 && cov <= 0.9) row("Game handicap", H + " " + sg(k), cov, q ? q.a : 0, k > 0 ? (fh ? "u" : "f") : (fh ? "f" : "u"), q ? "p" : "f");
    if (1 - cov >= 0.12 && 1 - cov <= 0.9) row("Game handicap", A + " " + sg(-k), 1 - cov, q ? q.b : 0, k > 0 ? (fh ? "f" : "u") : (fh ? "u" : "f"), q ? "p" : "f");
  }
  /* European (3-way) handicap, favourite gives h games: W1 / Draw / W2 plus the protected double chances */
  const F = fh ? H : A, D = fh ? A : H;
  for (let h = 1; h <= 4; h++) {
    let w1 = 0, x = 0, w2 = 0;
    for (let m = -40; m <= 40; m++) { const mf = fh ? m : -m, p = r.md[m + 40]; if (mf > h) w1 += p; else if (mf === h) x += p; else w2 += p; }
    const g = "European handicap " + h + ":0";
    row(g, F + " (W1, gives " + h + ")", w1, 0, "f", "f", 1);
    row(g, "Draw (wins by exactly " + h + ")", x, 0, "n", "f", 1);
    row(g, D + " (W2, gets " + h + ")", w2, 0, "u", "f", 1);
    row(g, F + " or Draw (1X)", w1 + x, 0, "f", "f", 1);
    row(g, D + " or Draw (X2)", w2 + x, 0, "u", "f", 1);
    row(g, "W1 or W2 (no draw)", w1 + w2, 0, "n", "f", 1);
  }
  return out;
}

function build(events, nowMs, generatedAt) {
  const skipped = {}, bump = (k) => { skipped[k] = (skipped[k] || 0) + 1; };
  const started = Date.now();

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
    if (Date.now() - started > 7000) { bump("out of time (will appear on the next refresh)"); return; }
    const L = c.lad, sig = c.ev.event_id + "|" + L.hLine + "|" + L.cover.toFixed(4) + "|" + L.tLine + "|" + L.pOver.toFixed(4);
    let sol = SOLVED.get(sig);
    if (!sol) { sol = solveMarket(L, TOUR_S[c.cat]); if (sol) SOLVED.set(sig, sol); }
    if (!sol) { bump("model did not fit the lines"); return; }
    const hur = { m: -L.hLine, t: L.tLine };
    const r = mix(sol.sA, sol.sB, sol.tau, hur);
    r.medTotal = medianTotal(r.gd);
    const pA = clamp(0.5 + (r.pA - 0.5) * CFG.CAL_GAMMA, 0.005, 0.995), pB = 1 - pA;
    const margin = -L.hdp, total = L.total;

    const real = mlIdx[c.home + "|" + c.away] || null;
    const est = [Math.max(1.01, 1 / (pA * (1 + CFG.MARGIN))), Math.max(1.01, 1 / (pB * (1 + CFG.MARGIN)))];
    const o = (real || est).map(R2);
    const ev = [pA * o[0] - 1, pB * o[1] - 1];
    const mlProb = real ? devig2(real[0], real[1]) : null;
    const pickem = o.every((x) => x >= CFG.PICKEM[0] && x <= CFG.PICKEM[1]);

    let side = pA >= pB ? 0 : 1, status = real ? "LEAN" : "MODEL";
    const gap = real ? Math.abs(pA - mlProb) : 0;                 // model vs Pinnacle moneyline, in probability
    if (real) {
      const b = ev[0] >= ev[1] ? 0 : 1;
      if (gap > 0.08) { side = mlProb >= 0.5 ? 0 : 1; status = "CHECK"; }   // big disagreement = distrust the model, follow Pinnacle
      else if (ev[b] > CFG.EV_EDGE) { side = b; status = "VALUE"; }
    }
    const name = side ? c.away : c.home, pp = side ? pB : pA, fav = pA >= pB ? 0 : 1, favName = fav ? c.away : c.home;
    let reason = "Lines imply " + c.home + " " + (margin >= 0 ? "-" : "+") + Math.abs(margin).toFixed(1) +
      " games over " + total.toFixed(1) + " total. Model: " + c.home + " " + R1(pA * 100) + "%, " + favName + " favoured.";
    reason += real ? " Pinnacle moneyline implies " + R1(mlProb * 100) + "% for " + c.home + "; model EV on the pick " + (ev[side] >= 0 ? "+" : "") + R1(ev[side] * 100) + "%."
                   : " No moneyline posted, so odds are estimated from the model with a " + R1(CFG.MARGIN * 100) + "% margin.";
    if (status === "CHECK") reason += " The model and the Pinnacle moneyline disagree by " + Math.round(gap * 100) + " points, so this is not treated as value; the pick follows the moneyline.";

    const markets = marketsFor(r, c.home, c.away, fav, L, real ? o : null);
    let pOverMkt = 0; for (let g = 0; g < r.gd.length; g++) if (g > L.tLine) pOverMkt += r.gd[g];
    const model = {
      fair: [1 / pA, 1 / pB].map(R2), book: est.map(R2),
      total: R1(r.eT), medTotal: r.medTotal == null ? null : R1(r.medTotal), spread: R1(r.eM), medSpread: (medianMargin(r.md) == null ? null : R1(medianMargin(r.md))),
      holds: [R1(hold(sol.sA) * 100), R1(hold(sol.sB) * 100)],
      mktTotal: R1(total), mktSpread: R1(margin), pOverMkt: R1(pOverMkt * 100), loose: Math.abs(sol.errT) > 0.03
    };
    const lg = String(c.ev.league_name), rd = (lg.match(ROUND_RE) || [])[1] || "";
    const pick = { model, odds: o, prediction: status, market: name + " Match Winner", pickOdds: o[side], prob: R1(pp * 100),
      reason, cat: c.cat, pickem, src: real ? "Pinnacle moneyline" : "estimated from model", markets };
    matches.push([c.home, c.away, new Date(c.st).toISOString(), [R1(pA * 100), R1(pB * 100)],
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

exports._test = { medianTotal, mix, hold, holdToPoint, tiebreak, setDist, matchStats, solveMarket, ladders, build, categoryOf };
