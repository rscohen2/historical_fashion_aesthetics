import { useMemo, useState } from 'react'
import TokenPicker from './TokenPicker.jsx'
import { Sparkline } from './Discover.jsx'
import { profilePartners } from './metrics.js'

const LIST_LENGTH = 30
const SORTS = [
  { id: 'count', label: 'Most frequent' },
  { id: 'lift', label: 'Most distinctive' },
]

const SOURCES = [
  { id: 'all', label: 'All' },
  { id: 'preset', label: 'Presets' },
  { id: 'yours', label: 'Yours' },
]

const pct = (v) => `${(100 * v).toFixed(v >= 0.1 ? 0 : v >= 0.01 ? 1 : 2)}%`
const times = (v) => `×${v.toFixed(v >= 10 ? 0 : 1)}`

// The terms an adjective (or group) describes most, or the adjectives (or
// adjective groups: presets and the user's own) most used on a term, over the
// selected decades and characters.
export default function TopPairs({
  data, kind, focus, gender, i0, i1, adjOptions, termOptions, groups,
  onFocus, onExamples, onPlot, plottedKeys,
}) {
  const [sort, setSort] = useState('count')
  const [minCount, setMinCount] = useState(20)
  const [partnerKind, setPartnerKind] = useState('adj') // term focus: 'adj' | 'group'
  const [source, setSource] = useState('all')
  const g = gender === 'split' ? 'all' : gender
  const showGroups = kind === 'term' && partnerKind === 'group'

  const shownGroups = useMemo(
    () => (source === 'all' ? groups : groups.filter((x) => x.source === source)),
    [groups, source],
  )
  const profile = useMemo(
    () => (focus ? profilePartners(data, focus, g, i0, i1, showGroups ? shownGroups : null) : null),
    [data, focus, g, i0, i1, showGroups, shownGroups],
  )
  const list = useMemo(() => {
    if (!profile) return []
    const rows = sort === 'lift'
      ? profile.partners.filter((p) => p.count >= minCount && p.lift != null)
      : [...profile.partners]
    rows.sort((a, b) => (sort === 'lift' ? b.lift - a.lift : b.count - a.count))
    return rows.slice(0, LIST_LENGTH)
  }, [profile, sort, minCount])

  const maxBar = list.length ? Math.max(...list.map((p) => (sort === 'lift' ? p.lift : p.share))) : 1
  const focusValue = focus ? (focus.kind === 'adj' ? focus.item.id : focus.term) : null
  const range = `${data.decades[i0]}s–${data.decades[i1]}s`

  return (
    <div className="at-discover">
      <div className="at-seg" role="tablist">
        <button type="button" className={kind === 'adj' ? 'active' : ''} onClick={() => onFocus({ kind: 'adj', id: undefined })}>
          Adjective → terms
        </button>
        <button type="button" className={kind === 'term' ? 'active' : ''} onClick={() => onFocus({ kind: 'term', id: undefined })}>
          Term → adjectives
        </button>
      </div>
      <TokenPicker
        label={kind === 'adj' ? 'Adjective' : 'Term'}
        placeholder={kind === 'adj' ? 'adjective or group…' : 'term…'}
        options={kind === 'adj' ? adjOptions : termOptions}
        selected={focusValue ? [focusValue] : []}
        onChange={(next) => onFocus({ kind, id: next[next.length - 1] ?? null })}
        compact
      />
      {kind === 'term' && (
        <div className="at-seg at-seg-sm" role="tablist" aria-label="Rank adjectives or groups">
          <button type="button" className={partnerKind === 'adj' ? 'active' : ''} onClick={() => setPartnerKind('adj')}>
            Adjectives
          </button>
          <button type="button" className={partnerKind === 'group' ? 'active' : ''} onClick={() => setPartnerKind('group')}>
            Adjective groups
          </button>
        </div>
      )}
      {showGroups && (
        <label className="at-side-control">
          Groups
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            {SOURCES.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
      )}
      {!profile ? (
        <p className="at-empty">Pick {kind === 'adj' ? 'an adjective or group' : 'a term'} to see its top {kind === 'adj' ? 'terms' : 'adjectives'}.</p>
      ) : (
        <>
          <div className="at-seg at-seg-sm" role="tablist">
            {SORTS.map((s) => (
              <button key={s.id} type="button" className={sort === s.id ? 'active' : ''} onClick={() => setSort(s.id)}>
                {s.label}
              </button>
            ))}
          </div>
          {sort === 'lift' && (
            <label className="at-side-control">
              Min. mentions
              <input type="number" min={0} step={10} value={minCount} onChange={(e) => setMinCount(Math.max(0, +e.target.value))} />
            </label>
          )}
          <p className="at-note">
            {profile.nFocus.toLocaleString()} {kind === 'adj' ? 'uses' : 'mentions'}, {range}
            {g !== 'all' ? ` (${g})` : ''}.{' '}
            {sort === 'count'
              ? <>Percent of {kind === 'adj' ? 'its uses that are on each term' : `its mentions using each ${showGroups ? 'group (members\' uses added up)' : 'adjective'}`}.</>
              : <>How many times more often each {kind === 'adj' ? 'term' : showGroups ? 'group' : 'adjective'} appears with it than across all fashion mentions.</>}
            {showGroups && ' Presets are WordNet categories (see Preset groups); big, loose ones rank high on count, so try Most distinctive.'}
            {' '}Click a row for examples; + plots the pair.
          </p>
          {list.length === 0 ? (
            <p className="at-empty">
              {showGroups && !shownGroups.length
                ? 'No groups yet: add a preset or group some adjectives above.'
                : `Nothing with at least ${minCount} mentions in this range.`}
            </p>
          ) : (
            <ol className="at-discover-list at-pairs-list">
              {list.map((p) => {
                const bar = sort === 'lift' ? p.lift : p.share
                return (
                  <li key={`${p.source ?? ''}:${p.spec.key}`}>
                    <button
                      type="button"
                      onClick={() => onExamples(p.pair)}
                      title={`${p.spec.item?.group ? `${p.spec.item.members.slice(0, 15).join(', ')}${p.spec.item.members.length > 15 ? ', …' : ''}\n` : ''}${p.count.toLocaleString()} mentions; ${pct(p.share)}; lift ${p.lift?.toFixed(2)}`}
                    >
                      <span className="at-d-label">
                        {p.spec.label}
                        {p.source === 'yours' && <span className="at-src-tag">yours</span>}
                      </span>
                      <span className="at-bar"><span style={{ width: `${(100 * bar) / maxBar}%` }} /></span>
                      <span className="at-d-change">{sort === 'lift' ? times(p.lift) : pct(p.share)}</span>
                      <span className="at-d-total">{p.count.toLocaleString()}</span>
                      <Sparkline values={p.spark} width={48} />
                    </button>
                    <button
                      type="button"
                      className="at-plot-btn"
                      disabled={plottedKeys.has(p.pair.key)}
                      onClick={() => onPlot(p.pair)}
                      aria-label={`Plot ${p.pair.label}`}
                      title={plottedKeys.has(p.pair.key) ? 'Already plotted' : `Plot “${p.pair.label}” over time`}
                    >
                      +
                    </button>
                  </li>
                )
              })}
            </ol>
          )}
        </>
      )}
    </div>
  )
}
