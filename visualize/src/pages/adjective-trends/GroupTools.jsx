import { useMemo, useState } from 'react'
import TokenPicker from './TokenPicker.jsx'
import { cleanGroupName } from './metrics.js'

const MAX_PRESETS = 60

// Searchable list of the preset groups (WordNet attribute categories).
export function PresetBrowser({ presets, adjTotals, onPick, onClose }) {
  const [query, setQuery] = useState('')

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const matches = q
      ? presets.filter((p) => p.name.includes(q) || p.definition.toLowerCase().includes(q) || p.members.includes(q))
      : presets
    return matches.slice(0, MAX_PRESETS)
  }, [presets, query])

  return (
    <div className="at-popover">
      <div className="at-popover-head">
        <strong>Preset adjective groups</strong>
        <button className="close-btn" type="button" onClick={onClose} aria-label="Close presets">✕</button>
      </div>
      <p className="at-note">
        WordNet attribute categories (from <code>adjective_categories.py</code>): an adjective joins
        every category one of its senses belongs to, so check the members. Adding a preset gives you
        an editable group. Search by name, definition, or member.
      </p>
      <input
        className="at-popover-search"
        value={query}
        placeholder="e.g. hue, size, beauty, shabby…"
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />
      {presets.length === 0 ? (
        <p className="at-empty">No presets in this export (WordNet wasn't available when it ran).</p>
      ) : (
        <ul className="at-preset-list">
          {rows.map((p) => (
            <li key={p.id}>
              <button type="button" onClick={() => onPick(p)}>
                <span className="at-preset-name">
                  {p.name}
                  <span className="at-count"> {p.members.length} adjectives · {p.members.reduce((a, m) => a + (adjTotals[m] ?? 0), 0).toLocaleString()} uses</span>
                </span>
                <span className="at-preset-def">{p.definition}</span>
                <span className="at-preset-members">{p.members.slice(0, 12).join(', ')}{p.members.length > 12 ? ', …' : ''}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// Rename a group, add or remove members, or break it up.
export function GroupEditor({ item, adjOptions, adjTotals, onChange, onUngroup, onDelete, onClose }) {
  const [name, setName] = useState(item.label)
  const commitName = () => {
    const clean = cleanGroupName(name)
    if (clean && clean !== item.label) onChange({ name: clean, members: item.members })
    else setName(item.label)
  }
  const total = item.members.reduce((a, m) => a + (adjTotals[m] ?? 0), 0)

  return (
    <div className="at-popover">
      <div className="at-popover-head">
        <strong>Edit group</strong>
        <button className="close-btn" type="button" onClick={onClose} aria-label="Close group editor">✕</button>
      </div>
      <label className="at-side-control">
        Name
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
      </label>
      <p className="at-note">
        {item.members.length} adjectives, {total.toLocaleString()} uses in total. A group&apos;s count adds up its
        members&apos; counts, so a mention with two members counts twice (this adds about 3.5% for the hue preset).
      </p>
      <TokenPicker
        label="Members"
        placeholder="add an adjective…"
        options={adjOptions}
        selected={item.members}
        onChange={(members) => members.length && onChange({ name: item.label, members })}
        compact
      />
      <div className="at-popover-actions">
        <button type="button" className="at-link" onClick={onUngroup}>Split into adjectives</button>
        <button type="button" className="at-link" onClick={onDelete}>Remove group</button>
        <button type="button" className="at-btn" onClick={onClose}>Done</button>
      </div>
    </div>
  )
}
