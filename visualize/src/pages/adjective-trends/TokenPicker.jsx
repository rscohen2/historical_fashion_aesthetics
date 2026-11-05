import { useMemo, useState } from 'react'

const MAX_SUGGESTIONS = 12

// Type-ahead multi-select. `options` is [{ value, label, count }] sorted by
// count (descending); matches on prefix first, then substring. Tokens whose
// value isn't an option (e.g. groups) are labelled by `tokenLabel`, and
// `onTokenClick` makes tokens clickable. `children` render after the box.
export default function TokenPicker({
  label, placeholder, options, selected, onChange, tokenLabel, onTokenClick, compact, children,
}) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)

  const byValue = useMemo(() => new Map(options.map((o) => [o.value, o])), [options])

  const suggestions = useMemo(() => {
    const q = query.trim().toLowerCase()
    const pool = options.filter((o) => !selected.includes(o.value))
    if (!q) return pool.slice(0, MAX_SUGGESTIONS)
    const prefix = []
    const inner = []
    for (const o of pool) {
      const hay = [o.value, o.label]
      if (hay.some((h) => h.startsWith(q))) prefix.push(o)
      else if (hay.some((h) => h.includes(q))) inner.push(o)
      if (prefix.length >= MAX_SUGGESTIONS) break
    }
    return [...prefix, ...inner].slice(0, MAX_SUGGESTIONS)
  }, [options, selected, query])

  const add = (value) => {
    onChange([...selected, value])
    setQuery('')
    setActive(0)
    setOpen(false)
  }

  const onKeyDown = (event) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setOpen(true)
      setActive((i) => Math.min(i + 1, suggestions.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActive((i) => Math.max(i - 1, 0))
    } else if (event.key === 'Enter' && suggestions[active]) {
      event.preventDefault()
      add(suggestions[active].value)
    } else if (event.key === 'Backspace' && !query && selected.length) {
      onChange(selected.slice(0, -1))
    } else if (event.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className={`at-picker ${compact ? 'compact' : ''}`}>
      {!compact && <span className="at-picker-label">{label}</span>}
      <div className="at-picker-box">
        {selected.map((value) => (
          <span key={value} className={`at-token ${value.startsWith('@') ? 'at-token-group' : ''}`}>
            {onTokenClick && value.startsWith('@') ? (
              <button type="button" className="at-token-label" onClick={() => onTokenClick(value)} title="Edit group">
                {tokenLabel?.(value) ?? byValue.get(value)?.label ?? value}
              </button>
            ) : (tokenLabel?.(value) ?? byValue.get(value)?.label ?? value)}
            <button
              type="button"
              aria-label={`Remove ${value}`}
              onClick={() => onChange(selected.filter((v) => v !== value))}
            >
              ✕
            </button>
          </span>
        ))}
        <div className="at-picker-input">
          <input
            value={query}
            placeholder={selected.length ? 'add…' : placeholder}
            onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0) }}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onKeyDown={onKeyDown}
            aria-label={label}
          />
          {open && suggestions.length > 0 && (
            <ul className="at-suggestions" role="listbox">
              {suggestions.map((o, i) => (
                <li
                  key={o.value}
                  role="option"
                  aria-selected={i === active}
                  className={i === active ? 'active' : ''}
                  // mousedown so the click lands before the input blurs
                  onMouseDown={(e) => { e.preventDefault(); add(o.value) }}
                  onMouseEnter={() => setActive(i)}
                >
                  <span>
                    {o.label}
                    {o.label !== o.value && !o.value.startsWith('@') && <span className="at-stem"> {o.value}</span>}
                  </span>
                  <span className="at-count">{o.count.toLocaleString()}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        {selected.length > 0 && !compact && (
          <button type="button" className="at-clear" onClick={() => onChange([])}>clear</button>
        )}
      </div>
      {children}
    </div>
  )
}
