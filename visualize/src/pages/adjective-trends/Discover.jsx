import { useMemo, useState } from 'react'
import * as d3 from 'd3'
import { discoverCandidates, rankCandidates, rankingMetric } from './metrics.js'

const LIST_LENGTH = 25
const SORTS = [
  { id: 'rising', label: 'Rising' },
  { id: 'falling', label: 'Falling' },
  { id: 'frequent', label: 'Most frequent' },
]
const FINDS = [
  { id: 'adj', label: 'Adjectives for selected terms' },
  { id: 'term', label: 'Terms for selected adjectives' },
  { id: 'pair', label: 'All pairs' },
]

export function Sparkline({ values, width = 72 }) {
  const w = width
  const h = 20
  const x = d3.scaleLinear().domain([0, values.length - 1]).range([1, w - 1])
  const y = d3.scaleLinear().domain([0, d3.max(values) || 1]).range([h - 2, 2])
  const line = d3.line().defined((v) => v != null).x((_, i) => x(i)).y((v) => y(v))
  return (
    <svg width={w} height={h} className="at-spark" aria-hidden="true">
      <path d={line(values)} />
    </svg>
  )
}

function formatChange(change) {
  const ratio = 2 ** change
  return ratio >= 1 ? `×${ratio.toFixed(ratio >= 10 ? 0 : 1)}` : `÷${(1 / ratio).toFixed(1 / ratio >= 10 ? 0 : 1)}`
}

export default function Discover({ data, mode, metric, gender, i0, i1, adjs, terms, selectedKeys, onAdd }) {
  const [sort, setSort] = useState('rising')
  const [minCount, setMinCount] = useState(50)
  const [pickedFind, setPickedFind] = useState(null)

  // In pair mode, default to ranking whatever dimension is not yet chosen.
  const find = pickedFind ?? (terms.length ? 'adj' : adjs.length ? 'term' : 'pair')
  const rankBy = rankingMetric(mode, metric)
  const rankGender = gender === 'split' ? 'all' : gender

  const ranked = useMemo(() => {
    const candidates = discoverCandidates(data, mode, find, adjs, terms)
    return rankCandidates(data, candidates, rankBy, rankGender, i0, i1, minCount)
  }, [data, mode, find, adjs, terms, rankBy, rankGender, i0, i1, minCount])

  const list = useMemo(() => {
    const sorted = [...ranked]
    if (sort === 'rising') sorted.sort((a, b) => b.change - a.change)
    else if (sort === 'falling') sorted.sort((a, b) => a.change - b.change)
    else sorted.sort((a, b) => b.total - a.total)
    return sorted.slice(0, LIST_LENGTH)
  }, [ranked, sort])

  const half = Math.floor((i0 + i1 + 1) / 2)
  const needsSelection = mode === 'pair' && ((find === 'adj' && !terms.length) || (find === 'term' && !adjs.length))

  return (
    <div className="at-discover">
      <div className="at-seg" role="tablist">
        {SORTS.map((s) => (
          <button key={s.id} type="button" className={sort === s.id ? 'active' : ''} onClick={() => setSort(s.id)}>
            {s.label}
          </button>
        ))}
      </div>
      {mode === 'pair' && (
        <label className="at-side-control">
          Rank
          <select value={find} onChange={(e) => setPickedFind(e.target.value)}>
            {FINDS.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
          </select>
        </label>
      )}
      <label className="at-side-control">
        Min. mentions
        <input type="number" min={0} step={10} value={minCount} onChange={(e) => setMinCount(Math.max(0, +e.target.value))} />
      </label>
      <p className="at-note">
        Change in <em>{rankBy.label.toLowerCase()}</em>{rankGender !== 'all' ? ` (${rankGender})` : ''},{' '}
        {data.decades[half]}s–{data.decades[i1]}s vs. {data.decades[i0]}s–{data.decades[half - 1] ?? data.decades[i0]}s.
        Click to add.
      </p>
      {needsSelection ? (
        <p className="at-empty">Select {find === 'adj' ? 'a term' : 'an adjective'} first.</p>
      ) : list.length === 0 ? (
        <p className="at-empty">Nothing with at least {minCount} mentions in this range.</p>
      ) : (
        <ol className="at-discover-list">
          {list.map(({ spec, total, change, spark }) => {
            const added = selectedKeys.has(spec.key)
            return (
              <li key={spec.key}>
                <button type="button" disabled={added} onClick={() => onAdd(spec)} title={added ? 'Already plotted' : 'Add to chart'}>
                  <span className="at-d-label">{spec.label}</span>
                  <Sparkline values={spark} />
                  <span className={`at-d-change ${change >= 0 ? 'up' : 'down'}`}>{formatChange(change)}</span>
                  <span className="at-d-total">{total.toLocaleString()}</span>
                </button>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
