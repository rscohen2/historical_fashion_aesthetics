import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import TokenPicker from './adjective-trends/TokenPicker.jsx'
import TrendChart from './adjective-trends/TrendChart.jsx'
import Discover from './adjective-trends/Discover.jsx'
import Examples from './adjective-trends/Examples.jsx'
import TopPairs from './adjective-trends/TopPairs.jsx'
import { PresetBrowser, GroupEditor } from './adjective-trends/GroupTools.jsx'
import {
  MODES, GENDER_OPTIONS, METRICS, MAX_SERIES, PALETTE,
  getMetric, buildSpecs, buildRows, assignColors, formatValue, rowsToCSV,
  adjItem, groupItem, parseAdjItems, serializeAdjItems, uniqueGroupName,
} from './adjective-trends/metrics.js'
import './adjective-trends/AdjectiveTrends.css'

const BASE_URL = `${import.meta.env.BASE_URL}data/adjective_trends/`

// Shown when the URL doesn't say otherwise.
const DEFAULTS = {
  mode: 'adj',
  a: 'white,black,grey,blue',
  t: 'dress,coat',
  g: 'all',
  y: 'linear',
  ci: '1',
  min: '30',
}

const Y_MODES = [
  { id: 'linear', label: 'Linear' },
  { id: 'log', label: 'Log' },
  { id: 'peak', label: 'Each series ÷ its peak' },
]

const splitList = (s) => (s ? s.split(',').filter(Boolean) : [])
const addItem = (list, item) => (list.some((i) => i.id === item.id) ? list : [...list, item])
const addTerm = (list, term) => (list.includes(term) ? list : [...list, term])
const sumCounts = (arr) => arr.reduce((a, b) => a + b, 0)

function DataTable({ rows, metric }) {
  const decades = rows[0]?.points.map((p) => p.decade) ?? []
  return (
    <div className="at-table-wrap">
      <table className="at-table">
        <thead>
          <tr>
            <th>Series</th>
            {decades.map((d) => <th key={d}>{d}s</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <th>
                <span className="at-swatch" style={{ background: row.color }} />
                {row.label}
              </th>
              {row.points.map((p) => (
                <td key={p.decade} title={p.d == null ? '' : `${p.k.toLocaleString()} / ${p.d.toLocaleString()}`}>
                  {formatValue(p.value, metric)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function AdjectiveTrends() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [params, setParams] = useSearchParams()
  const [colorMap, setColorMap] = useState({})
  // { key, decade, gender, spec? }: spec is set for series that aren't plotted.
  const [selected, setSelected] = useState(null)
  const [sideTab, setSideTab] = useState('discover')
  const [showTable, setShowTable] = useState(false)
  const [editingGroup, setEditingGroup] = useState(null) // group item id
  const [showPresets, setShowPresets] = useState(false)
  // Top pairs focus: { kind: 'adj' | 'term', id }. An undefined id follows the
  // chart's active or first series; null means the user cleared it.
  const [profileFocus, setProfileFocus] = useState({ kind: 'adj', id: undefined })

  useEffect(() => {
    fetch(`${BASE_URL}trends.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
        return r.json()
      })
      .then(setData)
      .catch((err) => setError(err.message))
  }, [])

  const get = (key) => params.get(key) ?? DEFAULTS[key]
  const mode = MODES.some((m) => m.id === get('mode')) ? get('mode') : DEFAULTS.mode
  const metric = getMetric(mode, params.get('m'))
  const gender = GENDER_OPTIONS.some((g) => g.id === get('g')) ? get('g') : DEFAULTS.g
  const yMode = Y_MODES.some((y) => y.id === get('y')) ? get('y') : DEFAULTS.y
  const showCI = get('ci') === '1'
  const minDen = Math.max(0, +get('min') || 0)
  const rawAdjs = get('a')
  const rawTerms = get('t')

  const update = useCallback((changes) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      for (const [key, value] of Object.entries(changes)) {
        if (value == null) next.delete(key)
        else next.set(key, value)
      }
      return next
    }, { replace: true })
  }, [setParams])

  // Selections, dropping anything the data doesn't know. Adjective-side
  // selections are items: single adjectives or groups.
  const items = useMemo(() => (data ? parseAdjItems(rawAdjs, data) : []), [data, rawAdjs])
  const terms = useMemo(
    () => (data ? splitList(rawTerms).filter((t) => data.terms[t]) : []),
    [data, rawTerms],
  )

  const nDecades = data?.decades.length ?? 0
  const fromIdx = data ? Math.max(0, data.decades.indexOf(+params.get('from'))) : 0
  const toParam = data ? data.decades.indexOf(+params.get('to')) : -1
  const toIdx = toParam > fromIdx ? toParam : Math.max(fromIdx + 1, nDecades - 1)

  const specs = useMemo(() => (data ? buildSpecs(data, mode, items, terms) : []), [data, mode, items, terms])
  const plotted = useMemo(() => specs.filter((s) => !s.missing).slice(0, MAX_SERIES), [specs])
  const colors = useMemo(() => assignColors(plotted, colorMap), [plotted, colorMap])
  const rows = useMemo(
    () => (data ? buildRows(data, plotted, metric, gender, fromIdx, toIdx, metric.den ? minDen : 0, colors) : []),
    [data, plotted, metric, gender, fromIdx, toIdx, minDen, colors],
  )
  const decades = useMemo(() => (data ? data.decades.slice(fromIdx, toIdx + 1) : []), [data, fromIdx, toIdx])
  const plottedKeys = useMemo(() => new Set(specs.map((s) => s.key)), [specs])
  const missing = specs.filter((s) => s.missing)
  const overflow = specs.filter((s) => !s.missing).length - plotted.length
  const plottedSpec = selected && plotted.find((s) => s.key === selected.key)
  const activeSpec = plottedSpec ?? selected?.spec ?? null
  const activeSelection = activeSpec ? selected : null

  // Commit a new selection, keeping colors stable for series that survive it.
  // `rename` ({ from, to } item ids) carries a renamed group's colors over.
  const setSelection = useCallback((next) => {
    const nextMode = next.mode ?? mode
    const nextItems = next.items ?? items
    const nextTerms = next.terms ?? terms
    const nextSpecs = buildSpecs(data, nextMode, nextItems, nextTerms).filter((s) => !s.missing).slice(0, MAX_SERIES)
    let previous = colors
    if (next.rename) {
      const { from, to } = next.rename
      previous = Object.fromEntries(Object.entries(colors).map(([key, slot]) => [
        key === `a:${from}` ? `a:${to}` : key.startsWith(`p:${from}|`) ? `p:${to}|${key.slice(from.length + 3)}` : key,
        slot,
      ]))
    }
    setColorMap((prev) => ({ ...prev, ...assignColors(nextSpecs, previous) }))
    update({
      mode: nextMode,
      a: serializeAdjItems(nextItems),
      t: nextTerms.join(','),
      ...(next.mode && next.mode !== mode ? { m: null } : {}),
    })
  }, [data, mode, items, terms, colors, update])

  const addFromDiscover = useCallback((spec) => {
    setSelection({
      items: spec.item ? addItem(items, spec.item) : items,
      terms: spec.term ? addTerm(terms, spec.term) : terms,
    })
  }, [items, terms, setSelection])

  // Put the pair's adjective and term first, so it is the first combination
  // plotted and never falls past the series cap.
  const plotPair = useCallback((spec) => {
    setSelection({
      mode: 'pair',
      items: [spec.item, ...items.filter((i) => i.id !== spec.item.id)],
      terms: [spec.term, ...terms.filter((t) => t !== spec.term)],
    })
  }, [items, terms, setSelection])

  const selectPoint = useCallback((sel) => {
    setSelected(sel)
    setSideTab('examples')
  }, [])

  const showProfile = useCallback((focus) => {
    setProfileFocus(focus)
    setSideTab('pairs')
  }, [])

  // Group editing. Groups live in the adjective selection itself.
  const replaceItem = (id, replacement) =>
    items.flatMap((i) => (i.id === id ? replacement : [i]))
  const addPreset = (preset) => {
    const members = preset.members.filter((m) => data.adjs[m])
    if (!members.length) return
    setSelection({ items: [...items, groupItem(uniqueGroupName(preset.name, items), members)] })
    setShowPresets(false)
  }
  const updateGroup = (id, { name, members }) => {
    const others = items.filter((i) => i.id !== id)
    const label = name === items.find((i) => i.id === id)?.label ? name : uniqueGroupName(name, others)
    const next = groupItem(label, members)
    setSelection({ items: replaceItem(id, [next]), rename: { from: id, to: next.id } })
    setEditingGroup(next.id)
  }
  const ungroup = (id) => {
    const group = items.find((i) => i.id === id)
    const singles = group.members.filter((m) => !items.some((i) => i.id === m)).map(adjItem)
    setSelection({ items: replaceItem(id, singles) })
    setEditingGroup(null)
  }
  const groupSelected = () => {
    const singles = items.filter((i) => !i.group)
    const base = singles.slice(0, 2).map((i) => i.label).join('-') + (singles.length > 2 ? ' etc' : '')
    const group = groupItem(uniqueGroupName(base, items), singles.map((i) => i.label))
    setSelection({ items: [...items.filter((i) => i.group), group] })
    setEditingGroup(group.id)
  }

  const adjTotals = useMemo(() => (data
    ? Object.fromEntries(Object.entries(data.adjs).map(([adj, rec]) => [adj, sumCounts(rec.n.all)]))
    : {}), [data])
  const adjOptions = useMemo(() => Object.entries(adjTotals)
    .map(([adj, count]) => ({ value: adj, label: adj, count }))
    .sort((a, b) => b.count - a.count), [adjTotals])
  // Top pairs can focus on a selected group as well as any adjective.
  const focusAdjOptions = useMemo(() => [
    ...items.filter((i) => i.group).map((i) => ({
      value: i.id, label: `${i.label} (group)`, count: i.members.reduce((a, m) => a + adjTotals[m], 0),
    })),
    ...adjOptions,
  ], [items, adjOptions, adjTotals])

  // Groups Top pairs can rank for a term: the user's groups, then the presets.
  // A preset with the same name as a user group is left out, since the user's
  // may be an edited copy of it.
  const rankGroups = useMemo(() => {
    if (!data) return []
    const yours = items.filter((i) => i.group)
    const presets = (data.groups ?? [])
      .filter((p) => !yours.some((i) => i.label === p.name))
      .map((p) => ({ item: groupItem(p.name, p.members.filter((m) => data.adjs[m])), source: 'preset' }))
      .filter((p) => p.item.members.length)
    return [...yours.map((item) => ({ item, source: 'yours' })), ...presets]
  }, [data, items])

  // Resolve the Top pairs focus, defaulting to the active or first series.
  const profile = useMemo(() => {
    if (!data) return null
    const { kind } = profileFocus
    let { id } = profileFocus
    if (id === undefined) {
      const source = [activeSpec, ...plotted].find((s) => s && (kind === 'term' ? s.term : s.item))
      id = source ? (kind === 'term' ? source.term : source.item.id) : null
    }
    if (id == null) return null
    if (kind === 'term') return data.terms[id] ? { kind, term: id } : null
    const item = items.find((i) => i.id === id) ?? (data.adjs[id] ? adjItem(id) : null)
    return item ? { kind, item } : null
  }, [data, profileFocus, activeSpec, plotted, items])
  const termOptions = useMemo(() => (data
    ? Object.entries(data.terms)
      .map(([stem, rec]) => ({ value: stem, label: rec.label, count: sumCounts(rec.n.all) }))
      .sort((a, b) => b.count - a.count)
    : []), [data])

  const downloadCSV = () => {
    const blob = new Blob([rowsToCSV(rows, metric)], { type: 'text/csv' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `adjective_trends_${mode}_${metric.id}_${gender}.csv`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  if (error) {
    return (
      <div className="page-header">
        <h1>Adjective Trends</h1>
        <p>Couldn't load <code>data/adjective_trends/trends.json</code> ({error}). Run{' '}
          <code>python -m fashion.analysis.adjective_trends</code> and check the{' '}
          <code>public/data</code> symlink.</p>
      </div>
    )
  }
  if (!data) return <div className="page-header"><p>Loading…</p></div>

  const editing = items.find((i) => i.id === editingGroup)
  const singlesSelected = items.filter((i) => !i.group).length
  const groupsNoBooks = metric.field === 'b' && plotted.some((s) => s.item?.group)
  const pickers = (
    <>
      {mode !== 'term' && (
        <div className="at-picker-wrap">
          <TokenPicker
            label="Adjectives"
            placeholder="e.g. shabby, gay, smart"
            options={adjOptions}
            selected={items.map((i) => i.id)}
            onChange={(ids) => setSelection({
              items: ids.map((id) => items.find((i) => i.id === id) ?? adjItem(id)),
            })}
            tokenLabel={(id) => {
              const item = items.find((i) => i.id === id)
              return item?.group ? `${item.label} · ${item.members.length}` : id
            }}
            onTokenClick={(id) => { setEditingGroup(id); setShowPresets(false) }}
          >
            <div className="at-picker-actions">
              <button type="button" className="at-btn" onClick={() => { setShowPresets((v) => !v); setEditingGroup(null) }}>
                Preset groups
              </button>
              {singlesSelected >= 2 && (
                <button type="button" className="at-btn" onClick={groupSelected} title="Combine the selected adjectives into one group">
                  Group selected
                </button>
              )}
            </div>
          </TokenPicker>
          {showPresets && (
            <PresetBrowser
              presets={data.groups ?? []}
              adjTotals={adjTotals}
              onPick={addPreset}
              onClose={() => setShowPresets(false)}
            />
          )}
          {editing && (
            <GroupEditor
              key={editing.id}
              item={editing}
              adjOptions={adjOptions}
              adjTotals={adjTotals}
              onChange={(next) => updateGroup(editing.id, next)}
              onUngroup={() => ungroup(editing.id)}
              onDelete={() => { setSelection({ items: items.filter((i) => i.id !== editing.id) }); setEditingGroup(null) }}
              onClose={() => setEditingGroup(null)}
            />
          )}
        </div>
      )}
      {mode !== 'adj' && (
        <TokenPicker
          label="Terms"
          placeholder="e.g. coat, bonnet, gown"
          options={termOptions}
          selected={terms}
          onChange={(next) => setSelection({ terms: next })}
        />
      )}
    </>
  )

  return (
    <div className="at-page">
      <div className="page-header">
        <h1>Adjective Trends</h1>
        <p>
          How the adjectives used to describe clothing, the clothing itself, and particular
          adjective + term combinations rise and fall across the corpus. Hover for values,
          click a point for example sentences, use <em>Discover</em> to find movers and
          <em> Top pairs</em> to see which terms an adjective goes with (or vice versa).
          Adjectives can be combined into groups, starting from WordNet presets. The URL
          keeps your view, so it can be bookmarked or shared.
        </p>
      </div>

      <div className="controls at-controls">
        <div className="at-seg" role="tablist" aria-label="What to plot">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              className={mode === m.id ? 'active' : ''}
              onClick={() => setSelection({ mode: m.id })}
            >
              {m.label}
            </button>
          ))}
        </div>
        <label>
          Measure
          <select value={metric.id} onChange={(e) => update({ m: e.target.value })}>
            {METRICS[mode].map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        <label>
          Characters
          <select value={gender} onChange={(e) => update({ g: e.target.value })}>
            {GENDER_OPTIONS.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
          </select>
        </label>
        <label>
          From
          <select value={data.decades[fromIdx]} onChange={(e) => update({ from: e.target.value })}>
            {data.decades.slice(0, toIdx).map((d) => <option key={d} value={d}>{d}s</option>)}
          </select>
        </label>
        <label>
          to
          <select value={data.decades[toIdx]} onChange={(e) => update({ to: e.target.value })}>
            {data.decades.slice(fromIdx + 1).map((d) => <option key={d} value={d}>{d}s</option>)}
          </select>
        </label>
      </div>

      <div className="at-pickers">{pickers}</div>

      <div className="at-layout">
        <div className="at-main">
          <div className="chart-container at-chart-card">
            <div className="at-chart-toolbar">
              <div className="at-legend">
                {plotted.map((spec) => (
                  <button
                    key={spec.key}
                    type="button"
                    className={`at-legend-item ${activeSelection?.key === spec.key ? 'active' : ''}`}
                    onClick={() => selectPoint({ key: spec.key, decade: null, gender: null })}
                    title="Show examples"
                  >
                    <span className="at-swatch" style={{ background: PALETTE[colors[spec.key]] }} />
                    {spec.label}
                  </button>
                ))}
                {gender === 'split' && plotted.length > 0 && (
                  <span className="at-linekey">
                    <svg width="22" height="8"><line x1="0" x2="22" y1="4" y2="4" /></svg> female
                    <svg width="22" height="8"><line x1="0" x2="22" y1="4" y2="4" strokeDasharray="6,4" /></svg> male
                  </span>
                )}
              </div>
              <div className="at-chart-options">
                <label>
                  Y axis
                  <select value={yMode} onChange={(e) => update({ y: e.target.value })}>
                    {Y_MODES.map((y) => <option key={y.id} value={y.id}>{y.label}</option>)}
                  </select>
                </label>
                {metric.den && (
                  <>
                    <label className="at-check">
                      <input type="checkbox" checked={showCI} onChange={(e) => update({ ci: e.target.checked ? '1' : '0' })} />
                      95% CI
                    </label>
                    <label title="Hide decades whose denominator is smaller than this">
                      Min. denominator
                      <input type="number" min={0} step={10} value={minDen} className="at-num" onChange={(e) => update({ min: e.target.value })} />
                    </label>
                  </>
                )}
              </div>
            </div>

            {plotted.length === 0 ? (
              <p className="at-empty at-empty-chart">
                {mode === 'pair'
                  ? 'Pick at least one adjective and one term. Every adjective × term combination is plotted.'
                  : `Pick ${mode === 'adj' ? 'adjectives' : 'terms'} above, or add some from Discover.`}
              </p>
            ) : (
              <TrendChart
                rows={rows}
                decades={decades}
                metric={metric}
                showCI={showCI}
                yMode={yMode}
                selected={activeSelection}
                onSelect={selectPoint}
              />
            )}

            {(missing.length > 0 || overflow > 0 || groupsNoBooks) && (
              <p className="at-note at-warn">
                {groupsNoBooks && '% of books isn\'t available for groups (book counts can\'t be added up). '}
                {overflow > 0 && `Showing the first ${MAX_SERIES} series; ${overflow} more are hidden — remove some to see them. `}
                {missing.length > 0 && (
                  <>Too rare to plot (fewer than {data.setup.min_pair_count} mentions in total):{' '}
                    {missing.map((s) => s.label).join(', ')}.</>
                )}
              </p>
            )}

            <div className="at-chart-footer">
              <span className="at-note">
                Counts are fashion mentions; an adjective counts once per mention it describes.
                {items.some((i) => i.group) && mode !== 'term' && ' Groups add up their members\' counts.'}
                {gender !== 'all' && ' Gender is the wearer\'s (gender_dep_pron); most mentions have none, so gendered views are smaller.'}
              </span>
              <span className="at-footer-actions">
                <button type="button" className="at-link" onClick={() => setShowTable((v) => !v)}>
                  {showTable ? 'Hide table' : 'Show table'}
                </button>
                <button type="button" className="at-link" onClick={downloadCSV} disabled={!rows.length}>
                  Download CSV
                </button>
              </span>
            </div>
            {showTable && rows.length > 0 && <DataTable rows={rows} metric={metric} />}
          </div>
        </div>

        <aside className="at-side">
          <div className="at-seg at-side-tabs" role="tablist">
            <button type="button" className={sideTab === 'discover' ? 'active' : ''} onClick={() => setSideTab('discover')}>
              Discover
            </button>
            <button type="button" className={sideTab === 'pairs' ? 'active' : ''} onClick={() => setSideTab('pairs')}>
              Top pairs
            </button>
            <button type="button" className={sideTab === 'examples' ? 'active' : ''} onClick={() => setSideTab('examples')}>
              Examples
            </button>
          </div>
          {/* Panels stay mounted so their settings survive tab switches. */}
          <div hidden={sideTab !== 'discover'}>
            <Discover
              data={data}
              mode={mode}
              metric={metric}
              gender={gender}
              i0={fromIdx}
              i1={toIdx}
              adjs={items}
              terms={terms}
              selectedKeys={plottedKeys}
              onAdd={addFromDiscover}
            />
          </div>
          <div hidden={sideTab !== 'pairs'}>
            <TopPairs
              data={data}
              kind={profileFocus.kind}
              focus={profile}
              gender={gender}
              i0={fromIdx}
              i1={toIdx}
              adjOptions={focusAdjOptions}
              termOptions={termOptions}
              groups={rankGroups}
              onFocus={setProfileFocus}
              onExamples={(spec) => selectPoint({ key: spec.key, spec, decade: null, gender: null })}
              onPlot={plotPair}
              plottedKeys={plottedKeys}
            />
          </div>
          {sideTab === 'examples' && (activeSpec ? (
            <Examples
              key={activeSpec.key}
              baseUrl={BASE_URL}
              data={data}
              spec={activeSpec}
              decade={activeSelection.decade}
              gender={gender === 'split' ? activeSelection.gender ?? 'split' : gender}
              onDecade={(decade) => setSelected({ ...activeSelection, decade })}
              onClose={() => setSelected(null)}
              onProfile={showProfile}
            />
          ) : (
            <p className="at-empty">Click a point on the chart, or a legend entry, to read example sentences.</p>
          ))}
        </aside>
      </div>
    </div>
  )
}
