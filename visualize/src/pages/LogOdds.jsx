import { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'

function highlightTerm(sentence, term) {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pattern = new RegExp(`(${escaped})`, 'gi')
  const parts = sentence.split(pattern)
  return parts.map((part, i) =>
    part.toLowerCase() === term.toLowerCase() ? <mark key={i}>{part}</mark> : part
  )
}

function SentencePanel({ term, adjective, sentences, onClose }) {
  return (
    <div className="lo-sentence-panel">
      <div className="lo-panel-header">
        <div>
          <span className="lo-term-label">{term}</span>
          <span className="lo-in"> — characters described as </span>
          <span className="lo-adj-label">"{adjective}"</span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <p className="sentence-count lo-hint">
        {sentences.length} passage{sentences.length !== 1 ? 's' : ''} mentioning <em>{term}</em>
        {' '}worn by characters described elsewhere as <em>{adjective}</em>
      </p>
      <div className="sentence-list">
        {sentences.map((s, i) => (
          <blockquote key={i} className="sentence">
            {highlightTerm(s, term)}
          </blockquote>
        ))}
      </div>
    </div>
  )
}

export default function LogOdds() {
  const [data, setData] = useState(null)          // raw JSON
  const [selectedTerm, setSelectedTerm] = useState('')
  const [topN, setTopN] = useState(20)
  const [scoreMode, setScoreMode] = useState('score') // 'score' | 'logodds'
  const [selectedAdj, setSelectedAdj] = useState(null)
  const chartRef = useRef(null)
  const tooltip = useRef(null)

  useEffect(() => {
    fetch('/data/logodds.json')
      .then((r) => r.json())
      .then((json) => {
        setData(json)
        setSelectedTerm(json.terms[0])
      })
  }, [])

  const selectAdj = useCallback((adj) => {
    setSelectedAdj((prev) => (prev === adj ? null : adj))
  }, [])

  // D3 chart
  useEffect(() => {
    if (!data || !selectedTerm) return
    const container = chartRef.current
    if (!container) return

    // Filter to selected term, rank by absolute value, then display ascending
    const termRows = data.logodds
      .filter((r) => r.term === selectedTerm)
      .sort((a, b) => Math.abs(b[scoreMode]) - Math.abs(a[scoreMode]))

    const top = termRows.slice(0, topN).sort((a, b) => a[scoreMode] - b[scoreMode])

    const margin = { top: 10, right: 80, bottom: 50, left: 120 }
    const rowH = 26
    const height = Math.max(300, top.length * rowH)
    const width = container.clientWidth - margin.left - margin.right

    d3.select(container).selectAll('*').remove()
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const xField = scoreMode
    const xExt = d3.extent(top, (d) => d[xField])
    const xPad = Math.max(Math.abs(xExt[0]), Math.abs(xExt[1]))
    const x = d3.scaleLinear()
      .domain([-xPad, xPad]).nice()
      .range([0, width])

    const y = d3.scaleBand()
      .domain(top.map((d) => d.adjective))
      .range([0, height])
      .padding(0.2)

    svg.append('g').attr('transform', `translate(0,${height})`).call(
      d3.axisBottom(x).ticks(6).tickFormat(d3.format('.2f'))
    )
    svg.append('g').attr('class', 'y-axis').call(d3.axisLeft(y))

    // Style y-axis tick labels to indicate selected adj and allow click
    svg.select('.y-axis').selectAll('.tick text')
      .style('cursor', 'pointer')
      .style('fill', (adj) => adj === selectedAdj ? 'var(--accent)' : null)
      .style('font-weight', (adj) => adj === selectedAdj ? '600' : null)
      .on('click', (_, adj) => selectAdj(adj))

    const zero = x(0)
    svg.append('line')
      .attr('x1', zero).attr('x2', zero)
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', '#d1d5db').attr('stroke-dasharray', '4,2')

    const colorPos = 'var(--accent)'
    const colorNeg = '#9ca3af'

    svg.selectAll('rect.bar')
      .data(top)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', (d) => d[xField] >= 0 ? zero : x(d[xField]))
      .attr('y', (d) => y(d.adjective))
      .attr('width', (d) => Math.abs(x(d[xField]) - zero))
      .attr('height', y.bandwidth())
      .attr('fill', (d) => d[xField] >= 0 ? colorPos : colorNeg)
      .attr('opacity', (d) => d.adjective === selectedAdj ? 1.0 : 0.7)
      .attr('stroke', (d) => d.adjective === selectedAdj ? 'var(--accent)' : 'none')
      .attr('stroke-width', 1.5)
      .attr('rx', 3)
      .style('cursor', 'pointer')
      .on('click', (_, d) => selectAdj(d.adjective))
      .on('mousemove', (event, d) => {
        const tip = tooltip.current
        if (!tip) return
        tip.style.display = 'block'
        tip.style.left = event.clientX + 12 + 'px'
        tip.style.top = event.clientY - 10 + 'px'
        const val = d[xField].toFixed(3)
        tip.textContent = `${d.adjective}: ${xField === 'score' ? 'z=' : 'δ='}${val} (n=${d.count.toLocaleString()})`
      })
      .on('mouseleave', () => { if (tooltip.current) tooltip.current.style.display = 'none' })

    // X axis label
    const xLabel = scoreMode === 'score'
      ? '← less associated   z-score (log-odds / σ)   more associated →'
      : '← less associated   log-odds (δ)   more associated →'
    svg.append('text')
      .attr('x', width / 2).attr('y', height + 42)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'var(--text-muted)')
      .text(xLabel)
  }, [data, selectedTerm, topN, scoreMode, selectedAdj, selectAdj])

  if (!data) {
    return <div className="page-header"><p>Loading…</p></div>
  }

  const sentences =
    selectedAdj && data.examples[selectedTerm]?.[selectedAdj]
      ? data.examples[selectedTerm][selectedAdj]
      : []

  return (
    <div>
      <div className="page-header">
        <h1>Character Adjective Log-Odds</h1>
        <p>
          Corpus-level Monroe et al. (2008) log-odds of character adjectives for each
          fashion term. Positive values (purple) mean the adjective is more characteristic
          of characters associated with that term; negative (grey) means less so.
          Click a bar or label to see example sentences.
        </p>
      </div>

      <div className="controls">
        <label>
          Fashion term:
          <select value={selectedTerm} onChange={(e) => { setSelectedTerm(e.target.value); setSelectedAdj(null) }}>
            {data.terms.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
        <label>
          Show top:
          <select value={topN} onChange={(e) => setTopN(+e.target.value)}>
            {[10, 15, 20, 30].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label>
          Metric:
          <select value={scoreMode} onChange={(e) => setScoreMode(e.target.value)}>
            <option value="score">z-score (δ/σ)</option>
            <option value="logodds">log-odds (δ)</option>
          </select>
        </label>
        <div className="lo-legend">
          <span className="lo-swatch lo-pos" />more associated
          <span className="lo-swatch lo-neg" style={{ marginLeft: '0.75rem' }} />less associated
        </div>
      </div>

      <div className="lo-layout">
        <div className="chart-container" style={{ overflowX: 'auto' }}>
          <div ref={chartRef} />
        </div>

        {selectedAdj && (
          <div className="lo-right-col">
            {sentences.length > 0 ? (
              <SentencePanel
                term={selectedTerm}
                adjective={selectedAdj}
                sentences={sentences}
                onClose={() => setSelectedAdj(null)}
              />
            ) : (
              <div className="lo-sentence-panel">
                <div className="lo-panel-header">
                  <span className="lo-adj-label">"{selectedAdj}"</span>
                  <button className="close-btn" onClick={() => setSelectedAdj(null)} aria-label="Close">✕</button>
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No example sentences available.</p>
              </div>
            )}
          </div>
        )}
      </div>

      <div ref={tooltip} className="tooltip" style={{ display: 'none' }} />

      <style>{`
        .lo-layout {
          display: grid;
          grid-template-columns: ${selectedAdj ? '1fr 340px' : '1fr'};
          gap: 1rem;
          align-items: start;
        }
        .lo-right-col {
          position: sticky;
          top: 60px;
          max-height: calc(100vh - 80px);
          overflow-y: auto;
        }
        .lo-sentence-panel {
          border: 1px solid var(--accent);
          border-radius: 8px;
          padding: 1.25rem;
          background: var(--accent-light);
        }
        .lo-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 0.5rem;
        }
        .lo-adj-label {
          font-weight: 600;
          color: var(--accent);
          font-size: 1rem;
        }
        .lo-in { color: var(--text-muted); margin: 0 0.3rem; }
        .lo-term-label { color: var(--text-h); font-size: 0.95rem; }
        .lo-legend {
          display: flex;
          align-items: center;
          gap: 0.3rem;
          font-size: 0.8rem;
          color: var(--text-muted);
        }
        .lo-swatch {
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 2px;
        }
        .lo-pos { background: var(--accent); }
        .lo-neg { background: #9ca3af; }

        /* reuse sentence styles from AdjectiveCategories */
        .sentence-count {
          font-size: 0.8rem;
          color: var(--text-muted);
          margin: 0 0 1rem;
          line-height: 1.5;
        }
        .lo-hint em { font-style: normal; color: var(--text-h); }
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
