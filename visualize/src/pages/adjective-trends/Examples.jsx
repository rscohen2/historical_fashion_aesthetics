import { useEffect, useMemo, useState } from 'react'
import { seriesCounts } from './metrics.js'
import BookCitation from '../../components/BookCitation.jsx'

// Example files are fetched once per session and shared across selections.
const cache = new Map()

function fetchExamples(baseUrl, kind, id) {
  const url = `${baseUrl}examples/${kind}/${id}.json`
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

// Split a span of text, wrapping whole-word matches of the mention's
// adjectives; those in `focus` (the plotted adjective or group) stand out.
function markAdjectives(text, focus, words, keyPrefix) {
  if (!words.length) return [text]
  const pattern = new RegExp(`(?<![\\p{L}])(${words.map(escapeRegExp).join('|')})(?![\\p{L}])`, 'giu')
  return text.split(pattern).map((part, i) => {
    if (i % 2 === 0) return part
    const isFocus = focus.has(part.toLowerCase())
    return <mark key={`${keyPrefix}${i}`} className={isFocus ? 'at-mark-adj' : 'at-mark-other'}>{part}</mark>
  })
}

function Sentence({ example, focus }) {
  const [start, end] = example.h
  return (
    <>
      {markAdjectives(example.s.slice(0, start), focus, example.a, 'pre')}
      <mark className="at-mark-term">{example.s.slice(start, end)}</mark>
      {markAdjectives(example.s.slice(end), focus, example.a, 'post')}
    </>
  )
}

// A group's examples come from its most-used members' files.
const MAX_GROUP_FILES = 6

// Which example files to load: [{ kind, id, adj }]. For an adjective item, the
// members with the most uses (in `decade`, or overall), skipping pairs too rare
// to have sampled examples.
function exampleSources(data, spec, decade) {
  if (spec.kind === 'term') return [{ kind: 'terms', id: data.terms[spec.term].id, adj: null }]
  const idx = data.decades.indexOf(decade)
  const sum = (arr) => (idx >= 0 ? arr[idx] : arr.reduce((a, b) => a + b, 0))
  const minPair = data.setup.min_pair_example_count ?? 0
  return spec.item.members
    .map((adj) => {
      const pair = spec.kind === 'pair' ? data.pairs[adj]?.[spec.term] : null
      if (spec.kind === 'pair' && (!pair || sum(pair.n.all) === 0 || pair.n.all.reduce((a, b) => a + b, 0) < minPair)) {
        return null
      }
      return { kind: 'adjs', id: data.adjs[adj].id, adj, weight: sum((pair ?? data.adjs[adj]).n.all) }
    })
    .filter((s) => s && s.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, MAX_GROUP_FILES)
}

// Interleave lists so each member contributes in turn.
function roundRobin(lists) {
  const out = []
  for (let i = 0; lists.some((l) => i < l.length); i++) {
    for (const list of lists) if (i < list.length) out.push(list[i])
  }
  return out
}

const GENDER_NAMES = { m: 'male', f: 'female' }

export default function Examples({ baseUrl, data, spec, decade, gender, onDecade, onClose, onProfile }) {
  const [state, setState] = useState({ url: null, payload: null, error: null })
  const [allGenders, setAllGenders] = useState(false)

  const sources = useMemo(() => exampleSources(data, spec, decade), [data, spec, decade])
  const url = sources.map((s) => `${s.kind}/${s.id}`).join(',')

  useEffect(() => {
    let live = true
    Promise.all(sources.map((s) => fetchExamples(baseUrl, s.kind, s.id)))
      .then((payload) => live && setState({ url, payload, error: null }))
      .catch((error) => live && setState({ url, payload: null, error: error.message }))
    return () => { live = false }
  }, [baseUrl, sources, url])

  const loading = state.url !== url
  // { decade: [examples] }, members interleaved for groups.
  const byDecade = useMemo(() => {
    if (loading || !state.payload) return {}
    const perSource = state.payload.map((payload) =>
      spec.kind === 'pair' ? payload.pairs?.[spec.term] ?? {} : payload.examples)
    return Object.fromEntries(data.decades.map((d) => [d, roundRobin(perSource.map((p) => p[d] ?? []))]))
  }, [loading, state.payload, spec, data])
  const focus = useMemo(() => new Set(spec.item?.members ?? []), [spec])

  const genderCode = gender === 'female' ? 'f' : gender === 'male' ? 'm' : null
  const pool = decade == null
    ? data.decades.flatMap((d) => byDecade[d] ?? [])
    : byDecade[decade] ?? []
  const matching = genderCode && !allGenders ? pool.filter((e) => e.g === genderCode) : pool

  const countGender = gender === 'split' ? 'all' : gender
  const counts = seriesCounts(data, spec, 'n', countGender)
  const decadeIdx = decade == null ? -1 : data.decades.indexOf(decade)
  const nMentions = counts
    ? decadeIdx >= 0 ? counts[decadeIdx] : counts.reduce((a, b) => a + b, 0)
    : 0

  return (
    <div className="at-examples">
      <div className="at-examples-head">
        <div>
          <div className="at-examples-title">{spec.label}</div>
          <div className="at-profile-links">
            {spec.item && (
              <button type="button" className="at-link" onClick={() => onProfile({ kind: 'adj', id: spec.item.id })}>
                Top terms for {spec.item.label} →
              </button>
            )}
            {spec.term && (
              <button type="button" className="at-link" onClick={() => onProfile({ kind: 'term', id: spec.term })}>
                Top adjectives for {data.terms[spec.term].label} →
              </button>
            )}
          </div>
          <div className="at-note">
            {nMentions.toLocaleString()} mention{nMentions === 1 ? '' : 's'}
            {countGender !== 'all' ? ` (${countGender})` : ''}
            {decade == null ? ' in total' : ` in the ${decade}s`}
          </div>
        </div>
        <button className="close-btn" type="button" onClick={onClose} aria-label="Close examples">✕</button>
      </div>
      <div className="at-examples-controls">
        <select value={decade ?? ''} onChange={(e) => onDecade(e.target.value === '' ? null : +e.target.value)}>
          <option value="">All decades</option>
          {data.decades.map((d) => <option key={d} value={d}>{d}s</option>)}
        </select>
        {genderCode && (
          <label className="at-check">
            <input type="checkbox" checked={allGenders} onChange={(e) => setAllGenders(e.target.checked)} />
            include other genders
          </label>
        )}
      </div>
      {loading && !state.error && <p className="at-empty">Loading examples…</p>}
      {state.error && state.url === url && <p className="at-empty">Couldn't load examples ({state.error}).</p>}
      {!loading && matching.length === 0 && (
        <p className="at-empty">
          No sampled examples{genderCode && !allGenders ? ` for ${gender} characters` : ''}
          {decade == null ? '' : ` in the ${decade}s`}.
          {spec.kind === 'pair' && !sources.length &&
            ` Examples are only sampled for pairs with at least ${data.setup.min_pair_example_count} mentions.`}
        </p>
      )}
      {!loading && matching.length > 0 && (
        <p className="at-note">
          Showing {matching.length} sampled example{matching.length === 1 ? '' : 's'}, spread across books
          {spec.item?.group && `, from ${sources.length < spec.item.members.length
            ? `the ${sources.length} most-used of ${spec.item.members.length} members`
            : `all ${sources.length} members`}`}.
          {' '}<mark className="at-mark-term">term</mark>{' '}
          {spec.kind !== 'term' && <><mark className="at-mark-adj">adjective</mark>{' '}</>}
          <mark className="at-mark-other">other adjectives</mark>
        </p>
      )}
      <div className="at-example-list">
        {!loading && matching.map((ex, i) => (
          <blockquote key={`${ex.bk}-${i}`} className="at-example">
            <Sentence example={ex} focus={focus} />
            <footer>
              <BookCitation id={ex.bk} />
              {ex.g && <span className="at-gender-tag">{GENDER_NAMES[ex.g]}</span>}
            </footer>
          </blockquote>
        ))}
      </div>
    </div>
  )
}
