import { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'

const RANKING_LABELS = {
  most_mentions: 'Most fashion mentions',
  most_distinct: 'Most distinct items',
  most_repeated: 'Most repeated item (mentions ÷ distinct)',
}

const RANKING_METRIC = {
  most_mentions: 'num_mentions',
  most_distinct: 'num_distinct',
  most_repeated: 'ratio',
}

function genderColor(gender) {
  if (gender === 'male') return 'var(--male)'
  if (gender === 'female') return 'var(--female)'
  return '#9ca3af'
}

function highlightTerm(sentence, term) {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pattern = new RegExp(`(${escaped})`, 'gi')
  const parts = sentence.split(pattern)
  return parts.map((part, i) =>
    part.toLowerCase() === term.toLowerCase() ? <mark key={i}>{part}</mark> : part
  )
}

function GenderBadge({ gender }) {
  const label = gender ? gender[0].toUpperCase() + gender.slice(1) : 'Unknown'
  return (
    <span className="ui-gender" style={{ background: genderColor(gender) }}>
      {label}
    </span>
  )
}

// One explorable character card. Reused for the ranking grid and for the
// scatter click-through, so it owns its own "which item is expanded" state.
const CHAR_ADJ_COLLAPSED = 8

function CharacterCard({ character, metric }) {
  const [openTerm, setOpenTerm] = useState(null)
  const [charAdjsOpen, setCharAdjsOpen] = useState(false)
  if (!character) return null

  const sentences = openTerm ? character.sentences?.[openTerm] ?? [] : []
  const charAdjs = charAdjsOpen
    ? character.char_adjs
    : character.char_adjs.slice(0, CHAR_ADJ_COLLAPSED)
  const hiddenCharAdjs = character.char_adjs.length - CHAR_ADJ_COLLAPSED

  return (
    <div className="ui-card">
      <div className="ui-card-head">
        <div className="ui-card-title">{character.title || '(untitled)'}</div>
        <div className="ui-badges">
          {character.is_narrator && <span className="ui-narrator">Narrator</span>}
          <GenderBadge gender={character.gender} />
        </div>
      </div>
      <div className="ui-card-author">{character.author || 'Unknown author'}</div>

      <div className="ui-stats">
        <span className={metric === 'num_mentions' ? 'ui-stat ui-stat-hi' : 'ui-stat'}>
          <b>{character.num_mentions}</b> mentions
        </span>
        <span className={metric === 'num_distinct' ? 'ui-stat ui-stat-hi' : 'ui-stat'}>
          <b>{character.num_distinct}</b> distinct
        </span>
        <span className={metric === 'ratio' ? 'ui-stat ui-stat-hi' : 'ui-stat'}>
          <b>{character.ratio}</b> per item
        </span>
      </div>

      <div className="ui-section-label">Clothing items <span className="ui-hint">(click for passages)</span></div>
      <div className="ui-chips">
        {character.top_clothes.map(({ term, count }) => {
          const has = (character.sentences?.[term]?.length ?? 0) > 0
          const isOpen = openTerm === term
          return (
            <button
              key={term}
              className={`ui-chip ui-chip-item${isOpen ? ' ui-chip-open' : ''}${has ? '' : ' ui-chip-empty'}`}
              onClick={() => setOpenTerm(isOpen ? null : term)}
              title={has ? 'Show passages' : 'No passages available'}
            >
              {term}
              {count > 1 && <span className="ui-chip-count">{count}</span>}
            </button>
          )
        })}
      </div>

      {openTerm && (
        <div className="ui-passages">
          <div className="ui-passages-head">
            <span>
              <em>{openTerm}</em> — {sentences.length} passage{sentences.length !== 1 ? 's' : ''}
            </span>
            <button className="close-btn" onClick={() => setOpenTerm(null)} aria-label="Close">✕</button>
          </div>
          {sentences.length > 0 ? (
            sentences.map((s, i) => (
              <blockquote key={i} className="sentence">{highlightTerm(s, openTerm)}</blockquote>
            ))
          ) : (
            <p className="ui-hint">No passages available for this item.</p>
          )}
        </div>
      )}

      {character.fashion_adjs.length > 0 && (
        <>
          <div className="ui-section-label">Fashion adjectives</div>
          <div className="ui-chips">
            {character.fashion_adjs.map(({ adj, count }) => (
              <span key={adj} className="ui-chip ui-chip-fashion">
                {adj}{count > 1 && <span className="ui-chip-count">{count}</span>}
              </span>
            ))}
          </div>
        </>
      )}

      {character.char_adjs.length > 0 && (
        <>
          <div className="ui-section-label">
            Character adjectives <span className="ui-hint">({character.char_adjs.length})</span>
          </div>
          <div className="ui-chips">
            {charAdjs.map((adj) => (
              <span key={adj} className="ui-chip ui-chip-char">{adj}</span>
            ))}
            {hiddenCharAdjs > 0 && (
              <button
                className="ui-chip ui-chip-toggle"
                onClick={() => setCharAdjsOpen((prev) => !prev)}
              >
                {charAdjsOpen ? 'Show less' : `+${hiddenCharAdjs} more`}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default function UniqueItems() {
  const [data, setData] = useState(null)
  const [ranking, setRanking] = useState('most_mentions')
  const [topN, setTopN] = useState(24)
  const [includeNarrators, setIncludeNarrators] = useState(false)
  const [selectedId, setSelectedId] = useState(null)
  const chartRef = useRef(null)
  const tooltip = useRef(null)

  useEffect(() => {
    fetch('/data/unique_items.json')
      .then((r) => r.json())
      .then(setData)
  }, [])

  const cards = useMemo(() => {
    if (!data) return []
    return data.rankings[ranking]
      .map((id) => data.characters[id])
      .filter((c) => c && (includeNarrators || !c.is_narrator))
      .slice(0, topN)
  }, [data, ranking, topN, includeNarrators])

  // D3 scatter: fashion mentions (x) vs distinct items (y).
  useEffect(() => {
    if (!data) return
    const container = chartRef.current
    if (!container) return

    const points = data.scatter
      .map((id) => data.characters[id])
      .filter((c) => c && (includeNarrators || !c.is_narrator))

    const margin = { top: 20, right: 20, bottom: 50, left: 55 }
    const width = container.clientWidth - margin.left - margin.right
    const height = 380

    d3.select(container).selectAll('*').remove()
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const maxX = d3.max(points, (d) => d.num_mentions) || 1
    const maxY = d3.max(points, (d) => d.num_distinct) || 1
    const x = d3.scaleLinear().domain([0, maxX * 1.05]).range([0, width])
    const y = d3.scaleLinear().domain([0, maxY * 1.05]).range([height, 0])

    svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(8))
    svg.append('g').call(d3.axisLeft(y).ticks(8))

    // y = x reference: distinct can never exceed mentions.
    const diagMax = Math.min(maxX, maxY)
    svg.append('line')
      .attr('x1', x(0)).attr('y1', y(0))
      .attr('x2', x(diagMax)).attr('y2', y(diagMax))
      .attr('stroke', '#d1d5db').attr('stroke-dasharray', '4,2')

    // Deterministic jitter so overlapping integer coordinates spread out.
    const jitter = (seed, spread) => ((Math.sin(seed * 99.7) + 1) / 2 - 0.5) * spread
    const xj = Math.min(6, width / maxX / 2)
    const yj = Math.min(6, height / maxY / 2)

    svg.selectAll('circle')
      .data(points)
      .join('circle')
      .attr('cx', (d, i) => x(d.num_mentions) + jitter(i + 1, xj))
      .attr('cy', (d, i) => y(d.num_distinct) + jitter(i + 7, yj))
      .attr('r', (d) => (d.id === selectedId ? 7 : 4))
      .attr('fill', (d) => genderColor(d.gender))
      .attr('opacity', (d) => (d.id === selectedId ? 1 : 0.55))
      .attr('stroke', (d) => (d.id === selectedId ? 'var(--text-h)' : '#fff'))
      .attr('stroke-width', (d) => (d.id === selectedId ? 2 : 1))
      .style('cursor', 'pointer')
      .on('mousemove', (event, d) => {
        const tip = tooltip.current
        if (!tip) return
        const clothes = d.top_clothes.slice(0, 4).map((c) => c.term).join(', ')
        tip.style.display = 'block'
        tip.style.left = event.clientX + 14 + 'px'
        tip.style.top = event.clientY - 10 + 'px'
        tip.innerHTML =
          `<div class="ui-tt-title">${d.title || '(untitled)'}</div>` +
          `<div class="ui-tt-sub">${d.author || 'Unknown'} · ${d.gender || 'unknown'}</div>` +
          `<div class="ui-tt-stats">${d.num_mentions} mentions · ${d.num_distinct} distinct · ${d.ratio}/item</div>` +
          `<div class="ui-tt-clothes">${clothes}</div>`
      })
      .on('mouseleave', () => { if (tooltip.current) tooltip.current.style.display = 'none' })
      .on('click', (_, d) => setSelectedId((prev) => (prev === d.id ? null : d.id)))

    svg.append('text')
      .attr('x', width / 2).attr('y', height + 42)
      .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', 'var(--text-muted)')
      .text('Fashion mentions')
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', -42)
      .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', 'var(--text-muted)')
      .text('Distinct fashion items')
  }, [data, selectedId, includeNarrators])

  if (!data) {
    return <div className="page-header"><p>Loading…</p></div>
  }

  const selected = selectedId ? data.characters[selectedId] : null

  return (
    <div>
      <div className="page-header">
        <h1>Unique Items</h1>
        <p>
          Which characters are the most fashion-y? Each card is a character, ranked by how
          many fashion mentions they have, how many distinct items they wear, or how often
          they repeat the same item. Click a clothing item to read the passages it appears
          in for that character.
        </p>
      </div>

      <div className="controls">
        <label>
          Rank by:
          <select value={ranking} onChange={(e) => setRanking(e.target.value)}>
            {Object.entries(RANKING_LABELS).map(([k, label]) => (
              <option key={k} value={k}>{label}</option>
            ))}
          </select>
        </label>
        <label>
          Show top:
          <select value={topN} onChange={(e) => setTopN(+e.target.value)}>
            {[12, 24, 48, 100].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label className="ui-check">
          <input
            type="checkbox"
            checked={includeNarrators}
            onChange={(e) => setIncludeNarrators(e.target.checked)}
          />
          Include narrators
        </label>
        <div className="ui-legend">
          <span className="ui-swatch" style={{ background: 'var(--male)' }} />male
          <span className="ui-swatch" style={{ background: 'var(--female)' }} />female
          <span className="ui-swatch" style={{ background: '#9ca3af' }} />unknown
        </div>
      </div>

      <div className="ui-card-grid">
        {cards.map((c) => (
          <CharacterCard key={c.id} character={c} metric={RANKING_METRIC[ranking]} />
        ))}
      </div>

      <div className="page-header" style={{ marginTop: '3rem' }}>
        <h2>Mentions vs. distinct items</h2>
        <p>
          Each dot is one of the most-mentioned characters
          {includeNarrators ? ' (narrators included)' : ' (narrators hidden)'}.
          Points sit on or below the diagonal because distinct items can never exceed total mentions.
          Hover for a summary; click a dot to pin its full card below.
        </p>
      </div>

      <div className="chart-container" style={{ overflowX: 'auto' }}>
        <div ref={chartRef} />
      </div>

      {selected && (
        <div className="ui-selected">
          <div className="ui-selected-head">
            <span>Selected character</span>
            <button className="close-btn" onClick={() => setSelectedId(null)} aria-label="Close">✕</button>
          </div>
          <div className="ui-card-grid ui-card-grid-single">
            <CharacterCard character={selected} metric={null} />
          </div>
        </div>
      )}

      <div ref={tooltip} className="tooltip" style={{ display: 'none' }} />

      <style>{`
        .ui-card-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 1rem;
          align-items: start;
        }
        .ui-card-grid-single { grid-template-columns: minmax(320px, 480px); }
        .ui-card {
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 1rem 1.1rem 1.1rem;
          background: var(--bg);
        }
        .ui-card-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 0.5rem;
        }
        .ui-card-title {
          font-weight: 600;
          color: var(--text-h);
          font-size: 0.95rem;
          line-height: 1.35;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .ui-card-author { font-size: 0.8rem; color: var(--text-muted); margin: 0.15rem 0 0.6rem; }
        .ui-badges { display: flex; gap: 0.3rem; flex-shrink: 0; align-items: center; }
        .ui-gender {
          flex-shrink: 0;
          color: #fff;
          font-size: 0.68rem;
          font-weight: 600;
          padding: 0.1rem 0.45rem;
          border-radius: 999px;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }
        .ui-narrator {
          flex-shrink: 0;
          color: var(--text-muted);
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          font-size: 0.68rem;
          font-weight: 600;
          padding: 0.1rem 0.45rem;
          border-radius: 999px;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }
        .ui-check { cursor: pointer; }
        .ui-check input { cursor: pointer; }
        .ui-stats { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 0.8rem; }
        .ui-stat { font-size: 0.8rem; color: var(--text-muted); }
        .ui-stat b { color: var(--text-h); font-size: 0.95rem; }
        .ui-stat-hi b { color: var(--accent); }
        .ui-section-label {
          font-size: 0.72rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--text-muted);
          margin: 0.7rem 0 0.4rem;
        }
        .ui-hint { font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--text-muted); font-size: 0.72rem; }
        .ui-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; }
        .ui-chip {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          font-size: 0.78rem;
          padding: 0.18rem 0.5rem;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: var(--bg-secondary);
          color: var(--text);
        }
        .ui-chip-count {
          background: rgba(0,0,0,0.08);
          border-radius: 999px;
          padding: 0 0.35rem;
          font-size: 0.68rem;
          font-weight: 600;
        }
        .ui-chip-item { cursor: pointer; }
        .ui-chip-item:hover { border-color: var(--accent); color: var(--accent); }
        .ui-chip-open { background: var(--accent); color: #fff; border-color: var(--accent); }
        .ui-chip-open .ui-chip-count { background: rgba(255,255,255,0.25); }
        .ui-chip-empty { opacity: 0.5; }
        .ui-chip-fashion { background: var(--accent-light); border-color: var(--accent-light); color: var(--accent); }
        .ui-chip-char { background: var(--bg-secondary); color: var(--text-muted); }
        .ui-chip-toggle { cursor: pointer; color: var(--accent); font-weight: 600; background: none; }
        .ui-chip-toggle:hover { border-color: var(--accent); }
        .ui-passages {
          margin: 0.6rem 0 0.2rem;
          padding: 0.75rem;
          border: 1px solid var(--accent);
          border-radius: 6px;
          background: var(--accent-light);
        }
        .ui-passages-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.82rem;
          color: var(--text-h);
          margin-bottom: 0.5rem;
        }
        .ui-passages-head em { font-style: normal; font-weight: 600; color: var(--accent); }
        .sentence {
          margin: 0 0 0.5rem;
          padding: 0.6rem 0.75rem;
          background: var(--bg);
          border: 1px solid var(--border);
          border-left: 3px solid var(--accent);
          border-radius: 4px;
          font-size: 0.82rem;
          line-height: 1.55;
          color: var(--text);
        }
        .sentence:last-child { margin-bottom: 0; }
        .sentence mark { background: #fef08a; color: #713f12; border-radius: 2px; padding: 0 1px; }
        .ui-legend { display: flex; align-items: center; gap: 0.3rem; font-size: 0.8rem; color: var(--text-muted); }
        .ui-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-left: 0.6rem; }
        .ui-legend .ui-swatch:first-child { margin-left: 0; }
        .ui-selected { margin-top: 1.25rem; }
        .ui-selected-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.8rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--text-muted);
          margin-bottom: 0.5rem;
        }
        .close-btn {
          background: none; border: none; cursor: pointer;
          font-size: 1rem; color: var(--text-muted); padding: 0 0.25rem; line-height: 1;
        }
        .close-btn:hover { color: var(--text-h); }
        .ui-tt-title { font-weight: 600; max-width: 260px; white-space: normal; }
        .ui-tt-sub { color: #d1d5db; font-size: 0.72rem; }
        .ui-tt-stats { margin-top: 0.2rem; }
        .ui-tt-clothes { color: #d1d5db; font-size: 0.72rem; max-width: 260px; white-space: normal; }
      `}</style>
    </div>
  )
}
