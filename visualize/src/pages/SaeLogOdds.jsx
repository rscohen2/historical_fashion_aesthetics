import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import * as d3 from 'd3'
import { toExample } from '../books.js'
import BookCitation from '../components/BookCitation.jsx'

const BASE_URL = `${import.meta.env.BASE_URL}data/sae_logodds/`

// Example files are fetched once per session and shared across selections.
const cache = new Map()

function fetchExamples(termIdx) {
  const url = `${BASE_URL}examples/${termIdx}.json`
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${r.status} loading ${url}`)
      return r.json()
    }).catch((err) => {
      cache.delete(url)
      throw err
    }))
  }
  return cache.get(url)
}

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

// Mark whole-word mentions of the term, plurals included.
function highlightTerm(sentence, term) {
  const pattern = new RegExp(`(?<![\\p{L}])(${escapeRegExp(term)}(?:e?s)?)(?![\\p{L}])`, 'giu')
  return sentence.split(pattern).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : part))
}

const codeName = (labels, code) => labels[code]?.label ?? `code ${code}`
const fullCodeName = (labels, code) =>
  labels[code] ? `${labels[code].label} (code ${code})` : `code ${code}`

function CodePanel({ term, termIdx, row, labels, related, onSelectTerm, onClose }) {
  const [state, setState] = useState({ termIdx: null, examples: null, error: null })

  useEffect(() => {
    let live = true
    fetchExamples(termIdx)
      .then((examples) => live && setState({ termIdx, examples, error: null }))
      .catch((err) => live && setState({ termIdx, examples: null, error: err.message }))
    return () => { live = false }
  }, [termIdx])

  const label = labels[row.code]
  const loading = state.termIdx !== termIdx
  const sentences = loading ? [] : (state.examples?.[row.code] ?? []).map(toExample)

  return (
    <div className="sl-panel">
      <div className="sl-panel-header">
        <div>
          <div className="sl-code-label">{label?.label ?? 'Unlabeled code'}</div>
          <div className="sl-code-meta">
            code {row.code} · z = {row.score.toFixed(2)} · n = {row.count.toLocaleString()}
          </div>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">✕</button>
      </div>
      {label?.description && <p className="sl-description">{label.description}</p>}
      <Link className="sl-trends-link" to={`/sae-code-trends?code=${row.code}`}>
        This code over time, by item →
      </Link>

      <h3 className="sl-subhead">
        Mentions of <em>{term}</em> where this code fires most strongly
      </h3>
      {loading && <p className="sl-empty">Loading examples…</p>}
      {state.error && !loading && <p className="sl-empty">Couldn't load examples ({state.error}).</p>}
      {!loading && !state.error && sentences.length === 0 && <p className="sl-empty">No examples.</p>}
      <div className="sentence-list">
        {sentences.map((ex, i) => (
          <blockquote key={i} className="sentence">
            {highlightTerm(ex.s, term)}
            <BookCitation id={ex.bk} />
          </blockquote>
        ))}
      </div>

      {related.length > 0 && (
        <>
          <h3 className="sl-subhead">Other items this code is associated with</h3>
          <div className="sl-related">
            {related.map(({ term: other, score }) => (
              <button key={other} type="button" className="sl-chip" onClick={() => onSelectTerm(other, row.code)}
                title={`z = ${score.toFixed(2)}`}>
                {other}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export default function SaeLogOdds() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [query, setQuery] = useState('')
  const [topN, setTopN] = useState(20)
  const [scoreMode, setScoreMode] = useState('score') // 'score' | 'logodds'
  const [params, setParams] = useSearchParams()
  const chartRef = useRef(null)
  const tooltip = useRef(null)

  useEffect(() => {
    fetch(`${BASE_URL}sae_logodds.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} loading sae_logodds.json`)
        return r.json()
      })
      .then(setData)
      .catch((err) => setError(err.message))
  }, [])

  const termIndex = useMemo(
    () => new Map((data?.terms ?? []).map((entry, i) => [entry.term, i])),
    [data],
  )
  // code → [{ term, score }] across every term's top codes, strongest first.
  const codeTerms = useMemo(() => {
    const byCode = new Map()
    for (const { term, codes } of data?.terms ?? []) {
      for (const { code, score } of codes) {
        if (!byCode.has(code)) byCode.set(code, [])
        byCode.get(code).push({ term, score })
      }
    }
    for (const list of byCode.values()) list.sort((a, b) => b.score - a.score)
    return byCode
  }, [data])

  const termParam = params.get('term')
  const termIdx = termIndex.has(termParam) ? termIndex.get(termParam) : 0
  const entry = data?.terms[termIdx]
  const selectedCode = params.has('code') ? +params.get('code') : null
  const selectedRow = entry?.codes.find((c) => c.code === selectedCode) ?? null

  // Keep `code` selected across terms when jumping from a code's related items.
  const selectTerm = (term, code) =>
    setParams(code == null ? { term } : { term, code: String(code) }, { replace: true })
  const selectCode = (code) => {
    const next = { term: entry.term }
    if (code !== selectedCode) next.code = String(code)
    setParams(next, { replace: true })
  }

  // D3 chart
  useEffect(() => {
    const container = chartRef.current
    if (!entry || !container) return
    const labels = data.labels
    // The bars under the cursor are about to be replaced, so mouseleave won't fire.
    if (tooltip.current) tooltip.current.style.display = 'none'

    const top = entry.codes
      .slice()
      .sort((a, b) => b[scoreMode] - a[scoreMode])
      .slice(0, topN)
    const keyOf = (d) => String(d.code)
    const truncate = (s) => (s.length > 34 ? `${s.slice(0, 33)}…` : s)

    const margin = { top: 10, right: 30, bottom: 50, left: 230 }
    const rowH = 26
    const height = Math.max(200, top.length * rowH)
    const width = Math.max(200, container.clientWidth - margin.left - margin.right)

    d3.select(container).selectAll('*').remove()
    if (!top.length) {
      d3.select(container).append('p').attr('class', 'sl-empty').text('No positively associated codes.')
      return
    }
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleLinear()
      .domain([0, d3.max(top, (d) => d[scoreMode])]).nice()
      .range([0, width])
    const y = d3.scaleBand()
      .domain(top.map(keyOf))
      .range([0, height])
      .padding(0.2)

    svg.append('g').attr('transform', `translate(0,${height})`).call(
      d3.axisBottom(x).ticks(6).tickFormat(d3.format('.2f'))
    )
    svg.append('g').attr('class', 'y-axis')
      .call(d3.axisLeft(y).tickFormat((key) => truncate(codeName(labels, key))))

    const isSelected = (key) => +key === selectedCode
    svg.select('.y-axis').selectAll('.tick text')
      .style('cursor', 'pointer')
      .style('fill', (key) => isSelected(key) ? 'var(--accent)' : null)
      .style('font-weight', (key) => isSelected(key) ? '600' : null)
      .on('click', (_, key) => selectCode(+key))
      .append('title').text((key) => fullCodeName(labels, key))

    svg.selectAll('rect.bar')
      .data(top)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', 0)
      .attr('y', (d) => y(keyOf(d)))
      .attr('width', (d) => x(d[scoreMode]))
      .attr('height', y.bandwidth())
      .attr('fill', 'var(--accent)')
      .attr('opacity', (d) => d.code === selectedCode ? 1.0 : 0.7)
      .attr('rx', 3)
      .style('cursor', 'pointer')
      .on('click', (_, d) => selectCode(d.code))
      .on('mousemove', (event, d) => {
        const tip = tooltip.current
        if (!tip) return
        tip.style.display = 'block'
        tip.style.left = event.clientX + 12 + 'px'
        tip.style.top = event.clientY - 10 + 'px'
        const val = d[scoreMode].toFixed(3)
        tip.textContent = `${fullCodeName(labels, d.code)}: ${scoreMode === 'score' ? 'z=' : 'δ='}${val} (n=${d.count.toLocaleString()})`
      })
      .on('mouseleave', () => { if (tooltip.current) tooltip.current.style.display = 'none' })

    svg.append('text')
      .attr('x', width / 2).attr('y', height + 42)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'var(--text-muted)')
      .text(scoreMode === 'score' ? 'z-score (log-odds / σ) →' : 'log-odds (δ) →')
  // selectCode only changes with entry/selectedCode, both already dependencies.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, entry, topN, scoreMode, selectedCode])

  if (error) {
    return (
      <div className="page-header">
        <p>Couldn't load data ({error}). Run <code>python -m fashion.analysis.sae_logodds</code> first.</p>
      </div>
    )
  }
  if (!data) {
    return <div className="page-header"><p>Loading…</p></div>
  }

  const q = query.trim().toLowerCase()
  const visibleTerms = q ? data.terms.filter(({ term }) => term.includes(q)) : data.terms
  const nLabeled = Object.keys(data.labels).length
  const related = selectedRow
    ? (codeTerms.get(selectedRow.code) ?? []).filter((r) => r.term !== entry.term).slice(0, 20)
    : []

  return (
    <div>
      <div className="page-header">
        <h1>SAE Code Log-Odds</h1>
        <p>
          Which sparse autoencoder codes, learned from sentence embeddings, are characteristic of
          the sentences mentioning each fashion item: Monroe et al. (2008) log-odds of each code
          for one item vs. all others. Codes are named by their LLM label. Pick an item on the
          left, and click a bar to see its description and the item's sentences where it fires.
        </p>
      </div>

      <div className="controls">
        <label>
          Show top:
          <select value={topN} onChange={(e) => setTopN(+e.target.value)}>
            {[10, 20, 30].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label>
          Metric:
          <select value={scoreMode} onChange={(e) => setScoreMode(e.target.value)}>
            <option value="score">z-score (δ/σ)</option>
            <option value="logodds">log-odds (δ)</option>
          </select>
        </label>
        {nLabeled === 0 && (
          <span className="sl-note">No code labels yet: run <code>sae_labels.py</code>, then rerun <code>sae_logodds.py</code>.</span>
        )}
      </div>

      <div className="sl-layout" style={{ gridTemplateColumns: selectedRow ? '220px 1fr 360px' : '220px 1fr' }}>
        <aside className="sl-terms">
          <input
            type="search"
            placeholder={`Search ${data.terms.length.toLocaleString()} items…`}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <ul>
            {visibleTerms.map(({ term, mentions }) => (
              <li key={term}>
                <button type="button" className={term === entry.term ? 'active' : ''} onClick={() => selectTerm(term)}>
                  <span>{term}</span>
                  <span className="sl-count">{mentions.toLocaleString()}</span>
                </button>
              </li>
            ))}
            {visibleTerms.length === 0 && <li className="sl-empty">No matching items.</li>}
          </ul>
        </aside>

        <div className="chart-container" style={{ overflowX: 'auto' }}>
          <h2 className="sl-term-title">
            {entry.term} <span className="sl-count">{entry.mentions.toLocaleString()} mentions</span>
          </h2>
          <div ref={chartRef} />
        </div>

        {selectedRow && (
          <div className="sl-right-col">
            <CodePanel
              term={entry.term}
              termIdx={termIdx}
              row={selectedRow}
              labels={data.labels}
              related={related}
              onSelectTerm={selectTerm}
              onClose={() => selectCode(selectedRow.code)}
            />
          </div>
        )}
      </div>

      <div ref={tooltip} className="tooltip" style={{ display: 'none' }} />

      <style>{`
        .sl-layout {
          display: grid;
          gap: 1rem;
          align-items: start;
        }
        .sl-terms, .sl-right-col {
          position: sticky;
          top: 60px;
          max-height: calc(100vh - 80px);
          overflow-y: auto;
        }
        .sl-terms {
          border: 1px solid var(--border);
          border-radius: 8px;
          background: var(--bg);
        }
        .sl-terms input {
          position: sticky;
          top: 0;
          width: 100%;
          box-sizing: border-box;
          padding: 0.5rem 0.75rem;
          border: none;
          border-bottom: 1px solid var(--border);
          background: var(--bg);
          font: inherit;
          font-size: 0.85rem;
        }
        .sl-terms ul { list-style: none; margin: 0; padding: 0.25rem 0; }
        .sl-terms button {
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
        .sl-terms button:hover { background: var(--bg-secondary); }
        .sl-terms button.active { background: var(--accent-light); color: var(--accent); font-weight: 600; }
        .sl-count { color: var(--text-muted); font-size: 0.75rem; font-weight: normal; }
        .sl-term-title { margin: 0 0 0.5rem; color: var(--text-h); }
        .sl-note { font-size: 0.8rem; color: var(--text-muted); }
        .sl-empty { color: var(--text-muted); font-size: 0.85rem; padding: 0 0.75rem; }
        .sl-panel {
          border: 1px solid var(--accent);
          border-radius: 8px;
          padding: 1.25rem;
          background: var(--accent-light);
        }
        .sl-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 0.5rem;
        }
        .sl-code-label { font-weight: 600; color: var(--accent); font-size: 1rem; }
        .sl-code-meta { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem; }
        .sl-trends-link { font-size: 0.8rem; color: var(--accent); text-decoration: none; }
        .sl-trends-link:hover { text-decoration: underline; }
        .sl-description { font-size: 0.85rem; color: var(--text-h); margin: 0 0 1rem; line-height: 1.5; }
        .sl-subhead {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--text-muted);
          margin: 1rem 0 0.5rem;
        }
        .sl-subhead em { font-style: normal; color: var(--text-h); }
        .sl-panel .sl-empty { padding: 0; }
        .sl-related { display: flex; flex-wrap: wrap; gap: 0.35rem; }
        .sl-chip {
          border: 1px solid var(--border);
          border-radius: 999px;
          background: var(--bg);
          padding: 0.15rem 0.6rem;
          font: inherit;
          font-size: 0.8rem;
          color: var(--text);
          cursor: pointer;
        }
        .sl-chip:hover { border-color: var(--accent); color: var(--accent); }
        .sentence-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .sentence {
          margin: 0;
          padding: 0.75rem 1rem;
          background: var(--bg);
          border: 1px solid var(--border);
          border-left: 3px solid var(--accent);
          border-radius: 4px;
          font-size: 0.85rem;
          line-height: 1.6;
          color: var(--text);
        }
        .sentence mark {
          background: #fef08a;
          color: #713f12;
          border-radius: 2px;
          padding: 0 1px;
        }
        .close-btn {
          background: none;
          border: none;
          cursor: pointer;
          font-size: 1rem;
          color: var(--text-muted);
          padding: 0 0.25rem;
          line-height: 1;
          flex-shrink: 0;
        }
        .close-btn:hover { color: var(--text-h); }
      `}</style>
    </div>
  )
}
