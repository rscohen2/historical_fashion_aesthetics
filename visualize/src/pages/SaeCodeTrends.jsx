import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import * as d3 from 'd3'
import { PALETTE, MAX_SERIES, formatValue, wilson } from './adjective-trends/metrics.js'
import TrendChart from './adjective-trends/TrendChart.jsx'

const BASE_URL = `${import.meta.env.BASE_URL}data/sae_logodds/`
const OTHER_COLOR = '#b6bcc6'
// Below this many mentions of an item in a decade, its "% of item's mentions" is too noisy to plot.
const MIN_ITEM_MENTIONS = 20

// `den` picks a point's denominator from {mentions, total, codeTotal,
// itemMentions}, and `factor`, if given, rescales the rate. The shape (axis,
// den, scale, suffix) is what TrendChart and formatValue expect.
const CODE_METRICS = [
  {
    id: 'rate', label: 'Per 1,000 fashion mentions', axis: 'activations per 1,000 fashion mentions',
    den: (ctx) => ctx.mentions, scale: 1000, suffix: '', stackable: true,
  },
  {
    id: 'within', label: '% of the item’s mentions', axis: '% of the item’s mentions that activate the code',
    den: (ctx) => ctx.itemMentions, minDen: MIN_ITEM_MENTIONS, scale: 100, suffix: '%', stackable: false,
  },
  {
    id: 'share', label: 'Share of the code’s activations', axis: '% of the code’s activations',
    den: (ctx) => ctx.total, scale: 100, suffix: '%', stackable: true,
  },
  {
    id: 'count', label: 'Raw count', axis: 'activations', den: null, scale: 1, suffix: '', stackable: true,
  },
]
const TERM_METRICS = [
  {
    id: 'within', label: '% of the item’s mentions', axis: '% of the item’s mentions that activate each code',
    den: (ctx) => ctx.itemMentions, minDen: MIN_ITEM_MENTIONS, scale: 100, suffix: '%',
  },
  {
    // The code's rate on the item ÷ its rate on all fashion mentions that decade.
    id: 'lift', label: 'Lift vs. all fashion mentions', axis: 'rate on the item ÷ rate on all fashion mentions',
    den: (ctx) => ctx.itemMentions, minDen: MIN_ITEM_MENTIONS, scale: 1, suffix: '×',
    factor: (ctx) => (ctx.codeTotal ? ctx.mentions / ctx.codeTotal : null),
  },
  {
    id: 'count', label: 'Raw count', axis: 'activations', den: null, scale: 1, suffix: '',
  },
]
const VIEWS = [
  { id: 'code', label: 'By code' },
  { id: 'term', label: 'By item' },
]
const CHARTS = [
  { id: 'lines', label: 'Lines' },
  { id: 'stacked', label: 'Stacked bars' },
]
const Y_MODES = [
  { id: 'linear', label: 'Linear' },
  { id: 'log', label: 'Log' },
  { id: 'peak', label: 'Each series ÷ its peak' },
]
const Z_OPTIONS = [0, 1.96, 3, 5, 10]

const NO_LABELS = {}
const codeName = (labels, code) => labels[code]?.label ?? `code ${code}`
const truncate = (s, n = 20) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

function fetchJson(name) {
  return fetch(`${BASE_URL}${name}`).then((r) => {
    if (!r.ok) throw new Error(`${r.status} loading ${name}`)
    return r.json()
  })
}

// Per-item trend files are fetched once per session and shared across selections.
const termCache = new Map()

function fetchTermTrends(idx) {
  if (!termCache.has(idx)) {
    termCache.set(idx, fetchJson(`term_trends/${idx}.json`).catch((err) => {
      termCache.delete(idx)
      throw err
    }))
  }
  return termCache.get(idx)
}

// A point as TrendChart draws it: value (scaled rate, or the raw count), its
// Wilson interval, and the k / d behind it.
function point(decade, k, ctx, metric) {
  if (!metric.den) return { decade, value: k, lo: null, hi: null, k, d: null }
  const d = metric.den(ctx)
  const scale = metric.factor ? metric.factor(ctx) * metric.scale : metric.scale
  if (!d || d < (metric.minDen ?? 1) || !scale) return { decade, value: null, lo: null, hi: null, k, d }
  // For lift, the all-mentions rate is treated as fixed: it rests on far more mentions.
  const [lo, hi] = wilson(k, d)
  return { decade, value: (k / d) * scale, lo: lo * scale, hi: hi * scale, k, d }
}

// One series per shown item, plus "other items". Items come sorted by z, and the
// filters only ever keep a prefix of them, so an item's color (its index) is stable.
function buildSeries(trends, entry, metric, minZ, maxItems) {
  const { decades, mentions, terms } = trends
  const shown = entry.items.filter((item) => item.score >= minZ).slice(0, maxItems)
  const series = shown.map((item, i) => ({
    spec: { key: item.term },
    label: item.term,
    color: PALETTE[i],
    points: decades.map((decade, d) => point(decade, item.n[d], {
      mentions: mentions[d], total: entry.total[d], itemMentions: terms[item.term]?.[d] ?? 0,
    }, metric)),
  }))
  const other = {
    spec: { key: '__other' },
    label: metric.id === 'within' ? 'all other items' : 'other items',
    color: OTHER_COLOR,
    dashed: true,
    points: decades.map((decade, d) => point(
      decade,
      entry.total[d] - d3.sum(shown, (item) => item.n[d]),
      {
        mentions: mentions[d],
        total: entry.total[d],
        itemMentions: mentions[d] - d3.sum(shown, (item) => terms[item.term]?.[d] ?? 0),
      },
      metric,
    )),
  }
  return { series, other }
}

// One series per shown code of an item, from its per-item trend file. As in
// buildSeries, the shown codes are a prefix of the item's z-sorted codes.
function buildTermSeries(trends, termEntry, termData, labels, metric, minZ, maxItems) {
  const { decades, mentions } = trends
  const shown = termEntry.codes.filter((row) => row.score >= minZ).slice(0, maxItems)
  return shown.map((row, i) => {
    const n = termData.codes[row.code] ?? decades.map(() => 0)
    const codeTotal = trends.codes[row.code]?.total
    const name = codeName(labels, row.code)
    return {
      spec: { key: String(row.code) },
      label: truncate(name),
      title: `${name} (code ${row.code})`,
      color: PALETTE[i],
      points: decades.map((decade, d) => point(decade, n[d], {
        mentions: mentions[d], codeTotal: codeTotal?.[d], itemMentions: termData.mentions[d],
      }, metric)),
    }
  })
}

function StackedChart({ series, decades, metric }) {
  const containerRef = useRef(null)
  const tipRef = useRef(null)
  const [width, setWidth] = useState(800)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(([e]) => setWidth(Math.max(320, e.contentRect.width)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const el = containerRef.current
    const tip = tipRef.current
    if (!el) return
    d3.select(el).selectAll('svg').remove()
    tip.style.display = 'none'
    if (!series.length) return

    const margin = { top: 12, right: 16, bottom: 36, left: 64 }
    const height = 400
    const innerW = width - margin.left - margin.right
    const innerH = height - margin.top - margin.bottom
    const x = d3.scaleBand().domain(decades).range([0, innerW]).padding(0.2)

    // Bottom-to-top in legend order, so "other" sits on top.
    const stacks = decades.map((decade, d) => {
      let y0 = 0
      return series.map((s) => {
        const v = s.points[d].value ?? 0
        const seg = { s, p: s.points[d], y0, y1: y0 + v }
        y0 += v
        return seg
      })
    })
    const yMax = d3.max(stacks, (col) => col[col.length - 1].y1)
    const y = d3.scaleLinear().domain([0, yMax || 1]).range([innerH, 0]).nice()

    const svg = d3.select(el).append('svg').attr('width', width).attr('height', height)
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    g.append('g').attr('class', 'sct-grid')
      .call(d3.axisLeft(y).ticks(6).tickSize(-innerW).tickFormat(''))
    g.append('g').attr('class', 'sct-axis').call(d3.axisLeft(y).ticks(6).tickSizeOuter(0))
    const tickEvery = Math.ceil(decades.length / Math.max(2, Math.floor(innerW / 56)))
    g.append('g').attr('class', 'sct-axis').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickSizeOuter(0)
        .tickValues(decades.filter((_, i) => i % tickEvery === 0))
        .tickFormat((d) => `${d}s`))
    g.append('text').attr('class', 'sct-axis-label')
      .attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -50)
      .attr('text-anchor', 'middle').text(metric.axis)

    const highlight = g.append('rect').attr('class', 'sct-highlight')
      .attr('y', 0).attr('height', innerH).attr('width', x.step()).style('display', 'none')

    // Segments separated by a thin surface-colored gap.
    g.append('g').selectAll('g').data(stacks).join('g')
      .selectAll('rect').data((col) => col.filter((seg) => seg.y1 > seg.y0)).join('rect')
      .attr('x', (seg) => x(seg.p.decade)).attr('width', x.bandwidth())
      .attr('y', (seg) => y(seg.y1)).attr('height', (seg) => Math.max(0, y(seg.y0) - y(seg.y1)))
      .attr('fill', (seg) => seg.s.color)
      .attr('stroke', 'var(--bg)').attr('stroke-width', 1)

    // Hover: one hit band per decade; the tooltip lists every series, largest first.
    g.append('g').selectAll('rect').data(decades).join('rect')
      .attr('x', (decade) => x(decade) - (x.step() - x.bandwidth()) / 2).attr('width', x.step())
      .attr('y', 0).attr('height', innerH).attr('fill', 'transparent')
      .on('pointermove', (event, decade) => {
        highlight.attr('x', x(decade) - (x.step() - x.bandwidth()) / 2).style('display', null)
        tip.replaceChildren()
        const head = document.createElement('div')
        head.className = 'sct-tip-head'
        head.textContent = `${decade}s`
        tip.appendChild(head)
        const rows = series.map((s) => ({ s, p: s.points.find((p) => p.decade === decade) }))
        rows.sort((a, b) => (b.p.value ?? -1) - (a.p.value ?? -1))
        for (const { s, p } of rows) {
          const row = document.createElement('div')
          row.className = 'sct-tip-row'
          const swatch = document.createElement('span')
          swatch.className = 'sct-swatch'
          swatch.style.background = s.color
          const value = document.createElement('strong')
          value.textContent = formatValue(p.value, metric)
          const label = document.createElement('span')
          label.textContent = s.label
          const counts = document.createElement('span')
          counts.className = 'sct-tip-counts'
          counts.textContent = p.d == null ? '' : `${p.k.toLocaleString()} / ${p.d.toLocaleString()}`
          row.append(swatch, value, label, counts)
          tip.appendChild(row)
        }
        tip.style.display = 'block'
        const box = tip.getBoundingClientRect()
        const left = event.clientX + 16 + box.width > window.innerWidth
          ? event.clientX - 16 - box.width : event.clientX + 16
        tip.style.left = `${left}px`
        tip.style.top = `${Math.max(8, Math.min(event.clientY - 10, window.innerHeight - box.height - 8))}px`
      })
      .on('pointerleave', () => {
        highlight.style('display', 'none')
        tip.style.display = 'none'
      })
  }, [series, decades, metric, width])

  return (
    <>
      <div ref={containerRef} className="sct-chart" />
      <div ref={tipRef} className="tooltip sct-tooltip" style={{ display: 'none' }} />
    </>
  )
}

export default function SaeCodeTrends() {
  const [trends, setTrends] = useState(null)
  const [logodds, setLogodds] = useState(null)
  const [termState, setTermState] = useState({ idx: null, data: null, error: null })
  const [error, setError] = useState(null)
  const [query, setQuery] = useState('')
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([fetchJson('trends.json'), fetchJson('sae_logodds.json')])
      .then(([t, lo]) => {
        setTrends(t)
        setLogodds(lo)
      })
      .catch((err) => setError(err.message))
  }, [])

  const labels = logodds?.labels ?? NO_LABELS
  const view = params.get('view') === 'term' ? 'term' : 'code'
  const metrics = view === 'term' ? TERM_METRICS : CODE_METRICS
  const metric = metrics.find((m) => m.id === params.get('metric')) ?? metrics[0]
  // Only the by-code view stacks, and shares and within-item rates don't stack at all.
  const chart = view === 'code' && metric.stackable && params.get('chart') === 'stacked' ? 'stacked' : 'lines'
  const yMode = Y_MODES.some((y) => y.id === params.get('y')) ? params.get('y') : 'linear'
  const showCI = params.get('ci') === '1'
  const minZ = params.has('z') ? +params.get('z') : 1.96
  const maxItems = Math.min(MAX_SERIES, Math.max(1, +(params.get('n') ?? 6)))
  // As a line, "other items" dwarfs the items on the per-1,000 and count scales,
  // so by default it is only drawn where it's a baseline or the stack's remainder.
  const showOther = params.has('other')
    ? params.get('other') !== '0'
    : chart === 'stacked' || metric.id === 'within'

  // Codes, most activations first.
  const codes = useMemo(() => {
    if (!trends) return []
    return Object.entries(trends.codes)
      .map(([code, entry]) => ({ code, entry, total: d3.sum(entry.total) }))
      .sort((a, b) => b.total - a.total)
  }, [trends])
  const termIndex = useMemo(
    () => new Map((logodds?.terms ?? []).map((entry, i) => [entry.term, i])),
    [logodds],
  )

  const code = trends?.codes[params.get('code')] ? params.get('code') : codes[0]?.code
  const entry = trends?.codes[code]
  const termIdx = termIndex.get(params.get('term')) ?? (logodds?.terms.length ? 0 : null)
  const termEntry = termIdx == null ? null : logodds.terms[termIdx]

  // The by-item view's counts are one small file per item, fetched on demand.
  useEffect(() => {
    if (view !== 'term' || termIdx == null) return
    let live = true
    fetchTermTrends(termIdx)
      .then((data) => live && setTermState({ idx: termIdx, data, error: null }))
      .catch((err) => live && setTermState({ idx: termIdx, data: null, error: err.message }))
    return () => { live = false }
  }, [view, termIdx])
  const termData = termState.idx === termIdx ? termState.data : null

  const update = (changes) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(changes)) next.set(k, String(v))
    setParams(next, { replace: true })
  }

  const { series, other } = useMemo(() => {
    if (view === 'term') {
      return termEntry && termData
        ? { series: buildTermSeries(trends, termEntry, termData, labels, metric, minZ, maxItems), other: null }
        : { series: [], other: null }
    }
    return entry ? buildSeries(trends, entry, metric, minZ, maxItems) : { series: [], other: null }
  }, [view, trends, entry, termEntry, termData, labels, metric, minZ, maxItems])
  // Stacked shares only add up to 100% with the remainder on top.
  const otherLocked = chart === 'stacked' && metric.id === 'share'
  const withOther = showOther || otherLocked
  const plotted = useMemo(() => (other && withOther ? [...series, other] : series), [series, other, withOther])

  // Clicking a line's point opens that (item, code)'s examples.
  const term = termEntry?.term
  const openExamples = useCallback(({ key }) => {
    if (key === '__other') return
    const [t, c] = view === 'term' ? [term, key] : [key, code]
    navigate(`/sae-log-odds?term=${encodeURIComponent(t)}&code=${c}`)
  }, [navigate, view, term, code])

  if (error) {
    return (
      <div className="page-header">
        <p>Couldn't load data ({error}). Run <code>python -m fashion.analysis.sae_logodds</code> first.</p>
      </div>
    )
  }
  if (!trends || !logodds) return <div className="page-header"><p>Loading…</p></div>

  const q = query.trim().toLowerCase()
  const sideList = view === 'code'
    ? codes
      .filter(({ code: c }) => !q || c === q || codeName(labels, c).toLowerCase().includes(q))
      .map(({ code: c, total }) => ({ key: c, name: codeName(labels, c), count: total, active: c === code }))
    : logodds.terms
      .filter(({ term: t }) => !q || t.includes(q))
      .map(({ term: t, mentions }) => ({ key: t, name: t, count: mentions, active: t === term }))
  const rows = view === 'code' ? entry.items : termEntry?.codes ?? []
  const qualifying = rows.filter((row) => row.score >= minZ).length
  const label = labels[code]
  const noun = view === 'code' ? 'items' : 'codes'
  const termLoading = view === 'term' && termEntry && !termData && !termState.error

  return (
    <div>
      <div className="page-header">
        <h1>SAE Code Trends</h1>
        <p>
          How SAE codes fire on fashion mentions decade by decade. <em>By code</em>: one code,
          broken down by the items it's most associated with (by log-odds z-score; see SAE
          Log-Odds); the rest are folded into <em>other items</em>. <em>By item</em>: one item,
          with a line for each of its most associated codes, to see which codes it leans on when.
          Hover a decade for counts; click a point, or a row below, for example sentences.
        </p>
      </div>

      <div className="controls">
        <div className="sct-seg" role="group" aria-label="View">
          {VIEWS.map((v) => (
            <button key={v.id} type="button" className={view === v.id ? 'active' : ''}
              onClick={() => update({ view: v.id })}>
              {v.label}
            </button>
          ))}
        </div>
        <label>
          Measure:
          <select value={metric.id} onChange={(e) => update({ metric: e.target.value })}>
            {metrics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        {view === 'code' && (
          <label>
            Chart:
            <select value={chart} onChange={(e) => update({ chart: e.target.value })}>
              {CHARTS.map((c) => (
                <option key={c.id} value={c.id} disabled={c.id === 'stacked' && !metric.stackable}>{c.label}</option>
              ))}
            </select>
          </label>
        )}
        {chart === 'lines' && (
          <>
            <label>
              Y axis:
              <select value={yMode} onChange={(e) => update({ y: e.target.value })}>
                {Y_MODES.map((y) => <option key={y.id} value={y.id}>{y.label}</option>)}
              </select>
            </label>
            {metric.den && (
              <label>
                <input type="checkbox" checked={showCI} onChange={(e) => update({ ci: e.target.checked ? 1 : 0 })} />
                95% CI
              </label>
            )}
          </>
        )}
        <label>
          Min z:
          <select value={minZ} onChange={(e) => update({ z: e.target.value })}>
            {Z_OPTIONS.map((z) => <option key={z} value={z}>{z === 0 ? 'any (z > 0)' : `≥ ${z}`}</option>)}
          </select>
        </label>
        <label>
          {view === 'code' ? 'Items' : 'Codes'}:
          <select value={maxItems} onChange={(e) => update({ n: e.target.value })}>
            {d3.range(1, MAX_SERIES + 1).map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        {view === 'code' && (
          <label title={otherLocked ? 'Stacked shares always include other items' : undefined}>
            <input type="checkbox" checked={withOther} disabled={otherLocked}
              onChange={(e) => update({ other: e.target.checked ? 1 : 0 })} />
            {metric.id === 'within' ? 'All other items (baseline)' : 'Other items'}
          </label>
        )}
      </div>

      <div className="sct-layout">
        <aside className="sct-codes">
          <input
            type="search"
            placeholder={view === 'code'
              ? `Search ${codes.length.toLocaleString()} codes…`
              : `Search ${logodds.terms.length.toLocaleString()} items…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <ul>
            {sideList.map(({ key, name, count, active }) => (
              <li key={key}>
                <button type="button" className={active ? 'active' : ''}
                  onClick={() => update(view === 'code' ? { code: key } : { term: key })}>
                  <span>{name}</span>
                  <span className="sct-count">{count.toLocaleString()}</span>
                </button>
              </li>
            ))}
            {sideList.length === 0 && <li className="sct-empty">No matching {noun === 'items' ? 'codes' : 'items'}.</li>}
          </ul>
        </aside>

        <div className="chart-container">
          {view === 'code' ? (
            <>
              <h2 className="sct-title">
                {label?.label ?? 'Unlabeled code'} <span className="sct-count">code {code}</span>
              </h2>
              {label?.description && <p className="sct-description">{label.description}</p>}
            </>
          ) : (
            <h2 className="sct-title">
              {term ?? 'No items'}{' '}
              {termEntry && <span className="sct-count">{termEntry.mentions.toLocaleString()} mentions</span>}
            </h2>
          )}

          <div className="sct-legend">
            {plotted.map((s) => {
              const hollow = s.dashed && chart === 'lines'
              return (
                <span key={s.spec.key} className="sct-legend-item" title={s.title}>
                  <span className={`sct-swatch${hollow ? ' dashed' : ''}`}
                    style={{ background: hollow ? 'none' : s.color, borderColor: s.color }} />
                  {s.label}
                </span>
              )
            })}
          </div>
          {termLoading && <p className="sct-empty">Loading…</p>}
          {view === 'term' && termState.error && termState.idx === termIdx && (
            <p className="sct-empty">Couldn't load this item's trends ({termState.error}).</p>
          )}
          {!termLoading && series.length === 0 && (
            <p className="sct-empty">
              No {noun} with z ≥ {minZ}{view === 'code' && withOther ? '; showing all items as other.' : '.'}
            </p>
          )}
          {chart === 'stacked' ? (
            <StackedChart series={plotted} decades={trends.decades} metric={metric} />
          ) : (
            // .at-page supplies the chart colors TrendChart's styles use.
            <div className="at-page">
              <TrendChart rows={plotted} decades={trends.decades} metric={metric} showCI={showCI}
                yMode={yMode} selected={null} onSelect={openExamples} />
            </div>
          )}
          {metric.minDen && (
            <p className="sct-note">
              Decades with fewer than {metric.minDen} mentions of an item are skipped (dotted).
              {metric.id === 'lift' && ' 1× means the code fires on this item as often as on fashion mentions overall.'}
            </p>
          )}

          <h3 className="sct-subhead">
            Most associated {noun} ({qualifying} of {rows.length} with z ≥ {minZ})
          </h3>
          <table className="sct-table">
            <thead>
              <tr><th /><th>{view === 'code' ? 'Item' : 'Code'}</th><th>z</th><th>Activations</th><th /></tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const key = view === 'code' ? row.term : String(row.code)
                const s = series.find((ser) => ser.spec.key === key)
                const [t, c] = view === 'code' ? [row.term, code] : [term, row.code]
                return (
                  <tr key={key} className={row.score < minZ ? 'muted' : ''}>
                    <td>
                      <span className="sct-swatch" style={{ background: s?.color ?? OTHER_COLOR, opacity: s ? 1 : 0.5 }} />
                    </td>
                    <td>{view === 'code' ? row.term : codeName(labels, row.code)}</td>
                    <td>{row.score.toFixed(1)}</td>
                    <td>{row.count.toLocaleString()}</td>
                    <td className="sct-links">
                      <Link to={`/sae-log-odds?term=${encodeURIComponent(t)}&code=${c}`}>examples</Link>
                      {view === 'code' ? (
                        termIndex.has(row.term) && (
                          <button type="button" onClick={() => update({ view: 'term', term: row.term })}>its codes</button>
                        )
                      ) : (
                        trends.codes[row.code] && (
                          <button type="button" onClick={() => update({ view: 'code', code: row.code })}>its items</button>
                        )
                      )}
                    </td>
                  </tr>
                )
              })}
              {rows.length === 0 && (
                <tr><td colSpan={5} className="sct-empty">No positively associated {noun}.</td></tr>
              )}
            </tbody>
          </table>
          <p className="sct-note">
            {view === 'code'
              ? `Only each code's top ${trends.setup.trend_terms} items are listed`
              : `Only each item's top ${logodds.setup.json_codes} codes are listed`}
            ; decades with fewer than {trends.setup.min_decade_mentions.toLocaleString()} dated
            fashion mentions are left out. The table's z and activations are over all mentions,
            dated or not.
          </p>
        </div>
      </div>

      <style>{`
        .sct-layout {
          display: grid;
          grid-template-columns: 240px minmax(0, 1fr);
          gap: 1rem;
          align-items: start;
        }
        @media (max-width: 720px) {
          .sct-layout { grid-template-columns: minmax(0, 1fr); }
          .sct-codes { position: static; max-height: 240px; }
        }
        .sct-codes {
          position: sticky;
          top: 60px;
          max-height: calc(100vh - 80px);
          overflow-y: auto;
          border: 1px solid var(--border);
          border-radius: 8px;
          background: var(--bg);
        }
        .sct-codes input {
          position: sticky;
          top: 0;
          width: 100%;
          padding: 0.5rem 0.75rem;
          border: none;
          border-bottom: 1px solid var(--border);
          border-radius: 0;
          background: var(--bg);
          font-size: 0.85rem;
        }
        .sct-codes ul { list-style: none; margin: 0; padding: 0.25rem 0; }
        .sct-codes button {
          display: flex;
          justify-content: space-between;
          gap: 0.5rem;
          width: 100%;
          padding: 0.3rem 0.75rem;
          border: none;
          background: none;
          font: inherit;
          font-size: 0.85rem;
          color: var(--text);
          text-align: left;
          cursor: pointer;
        }
        .sct-codes button:hover { background: var(--bg-secondary); }
        .sct-codes button.active { background: var(--accent-light); color: var(--accent); font-weight: 600; }
        .sct-count { color: var(--text-muted); font-size: 0.75rem; font-weight: normal; }
        .sct-title { margin: 0 0 0.25rem; }
        .sct-description { font-size: 0.85rem; margin: 0 0 0.75rem; }
        .sct-empty { color: var(--text-muted); font-size: 0.85rem; padding: 0 0.75rem; }
        .sct-note { font-size: 0.75rem; color: var(--text-muted); margin: 0.5rem 0 0; }
        .sct-legend { display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; font-size: 0.8rem; color: var(--text); margin-bottom: 0.5rem; }
        .sct-legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
        .sct-swatch {
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 2px;
          border: 1.5px solid transparent;
          flex-shrink: 0;
        }
        .sct-swatch.dashed { border-style: dashed; }
        .sct-chart { width: 100%; }
        .sct-chart svg { display: block; overflow: visible; }
        .sct-grid line { stroke: var(--border); }
        .sct-grid path { display: none; }
        .sct-axis { color: var(--text-muted); font-size: 11px; }
        .sct-axis path, .sct-axis line { stroke: var(--border); }
        .sct-axis-label { fill: var(--text-muted); font-size: 11px; }
        .sct-highlight { fill: var(--text-h); opacity: 0.05; pointer-events: none; }
        .sct-tooltip { white-space: normal; max-width: 340px; }
        .sct-tip-head { font-weight: 600; margin-bottom: 0.25rem; }
        .sct-tip-row { display: flex; align-items: center; gap: 0.4rem; }
        .sct-tip-row strong { min-width: 3.5em; text-align: right; font-variant-numeric: tabular-nums; }
        .sct-tip-counts { opacity: 0.7; margin-left: auto; padding-left: 0.75rem; font-variant-numeric: tabular-nums; }
        .sct-subhead { font-size: 0.85rem; color: var(--text-muted); margin: 1.5rem 0 0.5rem; }
        .sct-table { border-collapse: collapse; font-size: 0.85rem; width: 100%; max-width: 520px; }
        .sct-table th { text-align: left; font-weight: 600; color: var(--text-muted); font-size: 0.75rem; padding: 0.25rem 0.5rem; border-bottom: 1px solid var(--border); }
        .sct-table td { padding: 0.25rem 0.5rem; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
        .sct-table tr.muted td { color: var(--text-muted); }
        .sct-links { white-space: nowrap; }
        .sct-links a, .sct-links button {
          color: var(--accent);
          text-decoration: none;
          font: inherit;
          font-size: 0.8rem;
          background: none;
          border: none;
          padding: 0;
          margin-right: 0.75rem;
          cursor: pointer;
        }
        .sct-links a:hover, .sct-links button:hover { text-decoration: underline; }
        .sct-seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
        .sct-seg button {
          font: inherit;
          font-size: 0.85rem;
          border: none;
          background: var(--bg);
          color: var(--text);
          padding: 0.4rem 0.75rem;
          cursor: pointer;
        }
        .sct-seg button + button { border-left: 1px solid var(--border); }
        .sct-seg button.active { background: var(--accent-light); color: var(--accent); font-weight: 600; }
      `}</style>
    </div>
  )
}
