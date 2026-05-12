import { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'

function highlightTerm(sentence, term) {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = sentence.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === term.toLowerCase()
      ? <mark key={i}>{part}</mark>
      : part
  )
}

function SentencePanel({ category, term, sentences, onClose }) {
  return (
    <div className="sentence-panel">
      <div className="sentence-panel-header">
        <div>
          <span className="sentence-term">"{term}"</span>
          <span className="sentence-in"> in </span>
          <span className="sentence-category">{category.replace(/\.[a-z]\.\d+$/, '')}</span>
        </div>
        <button className="close-btn" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <p className="sentence-count">{sentences.length} example{sentences.length !== 1 ? 's' : ''}</p>
      <div className="sentence-list">
        {sentences.map((s, i) => (
          <blockquote key={i} className="sentence">
            {highlightTerm(s.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim(), term)}
          </blockquote>
        ))}
      </div>
    </div>
  )
}

export default function AdjectiveCategories() {
  const [data, setData] = useState([])       // [{category, count, label}]
  const [termMap, setTermMap] = useState({}) // category -> {definition, terms, example_sentences}
  const [minCount, setMinCount] = useState(100)
  const [selected, setSelected] = useState(null)   // row from data
  const [activeTerm, setActiveTerm] = useState(null)
  const chartRef = useRef(null)
  const tooltip = useRef(null)

  useEffect(() => {
    Promise.all([
      d3.csv('/data/category_counts.csv', (row) => ({
        category: row.category,
        count: +row.count,
        label: row.category.replace(/\.[a-z]\.\d+$/, ''),
      })),
      fetch('/data/category_to_term_map.jsonl')
        .then((r) => r.text())
        .then((text) =>
          text.trim().split('\n').reduce((acc, line) => {
            const obj = JSON.parse(line)
            acc[obj.category] = obj
            return acc
          }, {})
        ),
    ]).then(([counts, terms]) => {
      setData(counts)
      setTermMap(terms)
    })
  }, [])

  const selectCategory = useCallback((row) => {
    setSelected((prev) => prev?.category === row.category ? null : row)
    setActiveTerm(null)
  }, [])

  useEffect(() => {
    if (!data.length) return
    const container = chartRef.current
    if (!container) return

    const filtered = data
      .filter((d) => d.count >= minCount)
      .sort((a, b) => b.count - a.count)

    const margin = { top: 10, right: 80, bottom: 10, left: 160 }
    const rowH = 26
    const height = filtered.length * rowH
    const width = container.clientWidth - margin.left - margin.right

    d3.select(container).selectAll('*').remove()
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleLinear()
      .domain([0, d3.max(filtered, (d) => d.count)]).nice()
      .range([0, width])

    const y = d3.scaleBand()
      .domain(filtered.map((d) => d.category))
      .range([0, height]).padding(0.2)

    const baseColor = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, d3.max(filtered, (d) => d.count)])

    svg.append('g').call(
      d3.axisLeft(y).tickFormat((cat) => cat.replace(/\.[a-z]\.\d+$/, ''))
    )
    svg.selectAll('.tick text')
      .style('cursor', 'pointer')
      .style('fill', (cat) => selected?.category === cat ? 'var(--accent)' : null)
      .on('click', (_, cat) => {
        const row = filtered.find((d) => d.category === cat)
        if (row) selectCategory(row)
      })

    const rows = svg.selectAll('g.bar-row')
      .data(filtered)
      .join('g')
      .attr('class', 'bar-row')
      .style('cursor', 'pointer')
      .on('click', (_, d) => selectCategory(d))

    rows.append('rect')
      .attr('x', 0)
      .attr('y', (d) => y(d.category))
      .attr('width', (d) => x(d.count))
      .attr('height', y.bandwidth())
      .attr('fill', (d) => selected?.category === d.category ? 'var(--accent)' : baseColor(d.count))
      .attr('rx', 3)
      .on('mousemove', (event, d) => {
        const tip = tooltip.current
        if (!tip) return
        tip.style.display = 'block'
        tip.style.left = event.clientX + 12 + 'px'
        tip.style.top = event.clientY - 10 + 'px'
        tip.textContent = `${d.label}: ${d.count.toLocaleString()} — click to explore`
      })
      .on('mouseleave', () => { if (tooltip.current) tooltip.current.style.display = 'none' })

    rows.append('text')
      .attr('x', (d) => x(d.count) + 5)
      .attr('y', (d) => y(d.category) + y.bandwidth() / 2)
      .attr('dominant-baseline', 'middle')
      .attr('font-size', 11)
      .attr('fill', 'var(--text-muted)')
      .text((d) => d.count.toLocaleString())
  }, [data, minCount, selected, selectCategory])

  const info = selected ? termMap[selected.category] : null
  const sentences = activeTerm && info ? (info.example_sentences?.[activeTerm] ?? []) : []

  return (
    <div>
      <div className="page-header">
        <h1>Adjective Categories</h1>
        <p>
          Frequency of semantic adjective categories (WordNet synsets) used to describe fashion items.
          Click a bar to see adjectives, then click an adjective to see example sentences.
        </p>
      </div>

      <div className="controls">
        <label>
          Min. count:
          <select value={minCount} onChange={(e) => { setMinCount(+e.target.value); setSelected(null); setActiveTerm(null) }}>
            {[1, 10, 50, 100, 500, 1000].map((n) => (
              <option key={n} value={n}>{n.toLocaleString()}</option>
            ))}
          </select>
        </label>
        {data.length > 0 && (
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Showing {data.filter((d) => d.count >= minCount).length} of {data.length} categories
          </span>
        )}
      </div>

      <div className="adj-layout">
        <div className="chart-container" style={{ overflowX: 'auto' }}>
          <div ref={chartRef} />
        </div>

        {selected && info && (
          <div className="right-col">
            <div className="detail-panel">
              <div className="detail-header">
                <h2>{selected.label}</h2>
                <button className="close-btn" onClick={() => { setSelected(null); setActiveTerm(null) }} aria-label="Close">✕</button>
              </div>
              <p className="detail-synset">{selected.category}</p>
              {info.definition && (
                <p className="detail-def">{info.definition}</p>
              )}
              <p className="detail-count">{selected.count.toLocaleString()} occurrences</p>
              <p className="detail-hint">Click an adjective to see example sentences</p>
              <div className="term-list">
                {info.terms.map((term) => {
                  const hasExamples = info.example_sentences?.[term]?.length > 0
                  return (
                    <button
                      key={term}
                      className={`term-chip ${activeTerm === term ? 'active' : ''} ${!hasExamples ? 'no-examples' : ''}`}
                      onClick={() => setActiveTerm((prev) => prev === term ? null : term)}
                      disabled={!hasExamples}
                      title={hasExamples ? `${info.example_sentences[term].length} example(s)` : 'No examples'}
                    >
                      {term}
                      {hasExamples && (
                        <span className="term-count">{info.example_sentences[term].length}</span>
                      )}
                    </button>
                  )
                })}
              </div>
            </div>

            {activeTerm && sentences.length > 0 && (
              <SentencePanel
                category={selected.category}
                term={activeTerm}
                sentences={sentences}
                onClose={() => setActiveTerm(null)}
              />
            )}
          </div>
        )}
      </div>

      <div ref={tooltip} className="tooltip" style={{ display: 'none' }} />

      <style>{`
        .adj-layout {
          display: grid;
          grid-template-columns: ${selected ? '1fr 300px' : '1fr'};
          gap: 1rem;
          align-items: start;
        }
        .right-col {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          position: sticky;
          top: 60px;
          max-height: calc(100vh - 80px);
          overflow-y: auto;
        }
        .detail-panel {
          border: 1px solid var(--accent);
          border-radius: 8px;
          padding: 1.25rem;
          background: var(--accent-light);
        }
        .detail-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 0.25rem;
        }
        .detail-header h2 { margin: 0; color: var(--accent); }
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
        .detail-synset {
          font-size: 0.75rem;
          color: var(--text-muted);
          font-family: monospace;
          margin: 0 0 0.5rem;
        }
        .detail-def {
          font-size: 0.85rem;
          color: var(--text);
          font-style: italic;
          margin: 0 0 0.5rem;
          border-left: 3px solid var(--accent);
          padding-left: 0.5rem;
        }
        .detail-count {
          font-size: 0.8rem;
          color: var(--text-muted);
          margin: 0 0 0.25rem;
        }
        .detail-hint {
          font-size: 0.75rem;
          color: var(--text-muted);
          margin: 0 0 0.75rem;
        }
        .term-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
        }
        .term-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 0.2rem 0.5rem;
          font-size: 0.8rem;
          color: var(--text-h);
          cursor: pointer;
          font-family: inherit;
          transition: border-color 0.12s, background 0.12s;
        }
        .term-chip:hover:not(:disabled) {
          border-color: var(--accent);
          background: var(--accent-light);
          color: var(--accent);
        }
        .term-chip.active {
          background: var(--accent);
          border-color: var(--accent);
          color: #fff;
        }
        .term-chip.active .term-count { background: rgba(255,255,255,0.25); color: #fff; }
        .term-chip.no-examples { opacity: 0.4; cursor: default; }
        .term-count {
          font-size: 0.7rem;
          background: var(--border);
          color: var(--text-muted);
          border-radius: 3px;
          padding: 0 0.3rem;
          line-height: 1.4;
        }

        /* sentence panel */
        .sentence-panel {
          margin-top: 0;
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 1.25rem;
          background: var(--bg-secondary);
        }
        .sentence-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }
        .sentence-term {
          font-weight: 600;
          color: var(--accent);
          font-size: 1rem;
        }
        .sentence-in { color: var(--text-muted); margin: 0 0.3rem; }
        .sentence-category { color: var(--text-h); font-size: 0.95rem; }
        .sentence-count {
          font-size: 0.8rem;
          color: var(--text-muted);
          margin: 0 0 1rem;
        }
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
      `}</style>
    </div>
  )
}
