// Series construction, prevalence metrics, and change ranking for the
// Adjective Trends page. Everything here is pure so it can be memoized.

// Categorical palette (fixed order, validated for CVD separation). Color
// follows the series, never its rank: see assignColors.
export const PALETTE = [
  '#2a78d6', '#eb6834', '#1baf7a', '#eda100',
  '#e87ba4', '#008300', '#4a3aa7', '#e34948',
]
export const MAX_SERIES = PALETTE.length

export const MODES = [
  { id: 'adj', label: 'Adjectives' },
  { id: 'term', label: 'Terms' },
  { id: 'pair', label: 'Adjective + term' },
]

export const GENDER_OPTIONS = [
  { id: 'all', label: 'All characters' },
  { id: 'female', label: 'Female' },
  { id: 'male', label: 'Male' },
  { id: 'split', label: 'Female vs. male' },
]

// The adjective side of a series is an "item": a single adjective, or a named
// group of adjectives whose counts are added up. Items are serialized in the
// URL as `white` or `@name=red|blue|green`.
export function adjItem(adj) {
  return { id: adj, label: adj, members: [adj], group: false }
}

export function groupItem(name, members) {
  return { id: `@${name}`, label: name, members, group: true }
}

// Group names may not contain the URL delimiters (`,` `=` `|` `@`).
export function cleanGroupName(name) {
  return name.replace(/[^\p{L}\p{N} _'-]/gu, '').replace(/\s+/g, ' ').trim().slice(0, 40)
}

export function uniqueGroupName(name, items) {
  const base = cleanGroupName(name) || 'group'
  const taken = new Set(items.map((i) => i.id))
  let candidate = base
  for (let n = 2; taken.has(`@${candidate}`); n++) candidate = `${base} ${n}`
  return candidate
}

export function parseAdjItems(raw, data) {
  const items = []
  for (const token of raw ? raw.split(',') : []) {
    if (token.startsWith('@')) {
      const [name, list = ''] = token.slice(1).split('=')
      const members = [...new Set(list.split('|'))].filter((m) => data.adjs[m])
      if (name && members.length && !items.some((i) => i.id === `@${name}`)) {
        items.push(groupItem(name, members))
      }
    } else if (data.adjs[token] && !items.some((i) => i.id === token)) {
      items.push(adjItem(token))
    }
  }
  return items
}

export function serializeAdjItems(items) {
  return items.map((i) => (i.group ? `@${i.label}=${i.members.join('|')}` : i.id)).join(',')
}

function sumArrays(arrays) {
  if (arrays.length === 1) return arrays[0]
  const out = new Array(arrays[0].length).fill(0)
  for (const arr of arrays) for (let i = 0; i < arr.length; i++) out[i] += arr[i]
  return out
}

// Count arrays: field 'n' = mentions (adjective uses, for groups), 'b' =
// distinct books. Book counts can't be added up, so groups have none.
export function adjCounts(data, item, field, gender) {
  if (item.group && field === 'b') return null
  return sumArrays(item.members.map((m) => data.adjs[m][field][gender]))
}

export function pairCounts(data, item, term, field, gender) {
  if (item.group && field === 'b') return null
  const arrays = item.members.map((m) => data.pairs[m]?.[term]?.[field][gender]).filter(Boolean)
  return arrays.length ? sumArrays(arrays) : null
}

export function seriesCounts(data, spec, field, gender) {
  if (spec.kind === 'adj') return adjCounts(data, spec.item, field, gender)
  if (spec.kind === 'term') return data.terms[spec.term][field][gender]
  return pairCounts(data, spec.item, spec.term, field, gender)
}

export const adjSpec = (item) => ({ key: `a:${item.id}`, kind: 'adj', item, label: item.label })
export const termSpec = (data, term) => ({ key: `t:${term}`, kind: 'term', term, label: termLabel(data, term) })
export const pairSpec = (data, item, term, label) => ({
  key: `p:${item.id}|${term}`,
  kind: 'pair',
  item,
  term,
  label: label ?? `${item.label} ${termLabel(data, term)}`,
})

const totals = (key) => (data, spec, gender) => data.totals[gender][key]

// Each metric is k / d * scale per decade, with k and d count arrays. A null
// `den` means raw counts (no rate, no interval).
const PER_1K = {
  id: 'per1k', label: 'Per 1,000 fashion mentions', axis: 'per 1,000 fashion mentions',
  field: 'n', den: totals('mentions'), scale: 1000, suffix: '',
}
const BOOKS = {
  id: 'books', label: '% of books', axis: '% of books with a fashion mention',
  field: 'b', den: totals('books'), scale: 100, suffix: '%',
}
const COUNT = {
  id: 'count', label: 'Raw count', axis: 'mentions', field: 'n', den: null, scale: 1, suffix: '',
}

export const METRICS = {
  adj: [
    PER_1K,
    {
      id: 'described', label: '% of described mentions',
      axis: '% of mentions with any fashion adjective',
      field: 'n', den: totals('described'), scale: 100, suffix: '%',
    },
    BOOKS,
    COUNT,
  ],
  term: [PER_1K, BOOKS, COUNT],
  pair: [
    {
      id: 'ofTerm', label: '% of the term\'s mentions', axis: '% of the term\'s mentions using the adjective',
      field: 'n', den: (data, spec, gender) => data.terms[spec.term].n[gender], scale: 100, suffix: '%',
    },
    {
      id: 'ofAdj', label: '% of the adjective\'s uses', axis: '% of the adjective\'s uses on the term',
      field: 'n', den: (data, spec, gender) => adjCounts(data, spec.item, 'n', gender), scale: 100, suffix: '%',
    },
    { ...PER_1K, id: 'per10k', label: 'Per 10,000 fashion mentions', axis: 'per 10,000 fashion mentions', scale: 10000 },
    BOOKS,
    COUNT,
  ],
}

export function getMetric(mode, id) {
  return METRICS[mode].find((m) => m.id === id) ?? METRICS[mode][0]
}

// The metric used for ranking change: raw counts mostly track corpus size, so
// fall back to the mode's first (rate) metric.
export function rankingMetric(mode, metric) {
  return metric.den ? metric : METRICS[mode][0]
}

export function formatValue(value, metric) {
  if (value == null || Number.isNaN(value)) return '–'
  if (!metric.den) return Math.round(value).toLocaleString()
  const abs = Math.abs(value)
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3
  return value.toFixed(digits) + metric.suffix
}

// Wilson score interval for k successes out of n (as proportions).
export function wilson(k, n, z = 1.96) {
  if (!n) return [null, null]
  // Summed group counts can slightly exceed the denominator.
  const p = Math.min(k / n, 1)
  const z2 = z * z
  const denom = 1 + z2 / n
  const center = (p + z2 / (2 * n)) / denom
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom
  return [Math.max(0, center - half), Math.min(1, center + half)]
}

export function termLabel(data, stem) {
  return data.terms[stem]?.label ?? stem
}

// One spec per plotted entity. Pair mode plots every adjective-item × term
// combination; combinations below the export's pair threshold are `missing`.
export function buildSpecs(data, mode, items, terms) {
  if (mode === 'adj') return items.map(adjSpec)
  if (mode === 'term') return terms.map((term) => termSpec(data, term))
  return items.flatMap((item) =>
    terms.map((term) => ({
      ...pairSpec(data, item, term),
      missing: !item.members.some((m) => data.pairs[m]?.[term]),
    })),
  )
}

// Keep each surviving series' color slot; give new series the lowest free one.
export function assignColors(specs, previous) {
  const slots = {}
  const used = new Set()
  for (const { key } of specs) {
    if (previous[key] != null && !used.has(previous[key])) {
      slots[key] = previous[key]
      used.add(previous[key])
    }
  }
  for (const { key } of specs) {
    if (slots[key] != null) continue
    const free = PALETTE.findIndex((_, i) => !used.has(i))
    if (free === -1) break
    slots[key] = free
    used.add(free)
  }
  return slots
}

// Per-decade points for one series in one gender slice, over decade indices
// [i0, i1]. Decades whose denominator is below minDen get a null value.
export function seriesPoints(data, spec, metric, gender, i0, i1, minDen) {
  const k = seriesCounts(data, spec, metric.field, gender)
  if (!k) return null
  const d = metric.den ? metric.den(data, spec, gender) : null
  const points = []
  for (let i = i0; i <= i1; i++) {
    const decade = data.decades[i]
    if (!d) {
      points.push({ decade, value: k[i], lo: null, hi: null, k: k[i], d: null })
      continue
    }
    if (!d[i] || d[i] < minDen) {
      points.push({ decade, value: null, lo: null, hi: null, k: k[i], d: d[i] })
      continue
    }
    const [lo, hi] = wilson(k[i], d[i])
    points.push({
      decade,
      value: (k[i] / d[i]) * metric.scale,
      lo: lo * metric.scale,
      hi: hi * metric.scale,
      k: k[i],
      d: d[i],
    })
  }
  return points
}

// Rows to plot: one per series, or two (female solid, male dashed) in split mode.
export function buildRows(data, specs, metric, genderMode, i0, i1, minDen, colors) {
  const genders = genderMode === 'split' ? ['female', 'male'] : [genderMode]
  const rows = []
  for (const spec of specs) {
    if (colors[spec.key] == null) continue
    for (const gender of genders) {
      const points = seriesPoints(data, spec, metric, gender, i0, i1, minDen)
      if (!points) continue
      rows.push({
        id: `${spec.key}#${gender}`,
        spec,
        gender,
        color: PALETTE[colors[spec.key]],
        dashed: genderMode === 'split' && gender === 'male',
        label: genderMode === 'split' ? `${spec.label} (${gender})` : spec.label,
        points,
      })
    }
  }
  return rows
}

// Candidate series for the Discover panel. In pair mode `find` picks what is
// being ranked: adjectives for the selected terms, terms for the selected
// adjectives, or all pairs.
export function discoverCandidates(data, mode, find, items, terms) {
  if (mode === 'adj') return Object.keys(data.adjs).map((adj) => adjSpec(adjItem(adj)))
  if (mode === 'term') return Object.keys(data.terms).map((term) => termSpec(data, term))
  if (find === 'adj') {
    return Object.entries(data.pairs).flatMap(([adj, byTerm]) =>
      terms.filter((t) => byTerm[t]).map((t) =>
        pairSpec(data, adjItem(adj), t, terms.length > 1 ? `${adj} (${termLabel(data, t)})` : adj)),
    )
  }
  if (find === 'term') {
    return items.flatMap((item) =>
      itemTerms(data, item).map((t) =>
        pairSpec(data, item, t, items.length > 1 ? `${termLabel(data, t)} (${item.label})` : termLabel(data, t))),
    )
  }
  return Object.entries(data.pairs).flatMap(([adj, byTerm]) =>
    Object.keys(byTerm).map((t) => pairSpec(data, adjItem(adj), t)),
  )
}

// Terms that any member of an item has pair counts with.
export function itemTerms(data, item) {
  return [...new Set(item.members.flatMap((m) => Object.keys(data.pairs[m] ?? {})))]
}

// Adjectives with pair counts for each term, built once per dataset.
const termIndexCache = new WeakMap()
export function termAdjectives(data, term) {
  if (!termIndexCache.has(data)) {
    const index = {}
    for (const [adj, byTerm] of Object.entries(data.pairs)) {
      for (const t of Object.keys(byTerm)) (index[t] ??= []).push(adj)
    }
    termIndexCache.set(data, index)
  }
  return termIndexCache.get(data)[term] ?? []
}

const rangeSum = (arr, i0, i1) => {
  let total = 0
  for (let i = i0; i <= i1; i++) total += arr[i]
  return total
}

// Top partners of a focus: the terms an adjective item describes, or the
// adjectives describing a term, over decades [i0, i1]. For each partner:
//   count  pair mentions
//   share  count / the focus's mentions (adjective uses, for an item)
//   lift   share / the partner's overall rate per fashion mention, i.e. how
//          many times more often the two go together than the partner's
//          baseline would suggest (symmetric in focus and partner).
// For a term focus, `groups` ([{ item, source }]) ranks those adjective groups
// as the partners instead of single adjectives; counts are summed as usual.
export function profilePartners(data, focus, gender, i0, i1, groups = null) {
  const N = rangeSum(data.totals[gender].mentions, i0, i1)
  const focusCounts = focus.kind === 'adj'
    ? adjCounts(data, focus.item, 'n', gender)
    : data.terms[focus.term].n[gender]
  const nFocus = rangeSum(focusCounts, i0, i1)
  const partners = focus.kind === 'adj'
    ? itemTerms(data, focus.item).map((term) => ({
      spec: termSpec(data, term),
      pair: pairSpec(data, focus.item, term),
      k: pairCounts(data, focus.item, term, 'n', gender),
      partnerCounts: data.terms[term].n[gender],
    }))
    : groups
      ? groups.map(({ item, source }) => ({
        spec: adjSpec(item),
        pair: pairSpec(data, item, focus.term),
        source,
        k: pairCounts(data, item, focus.term, 'n', gender),
        partnerCounts: adjCounts(data, item, 'n', gender),
      })).filter((p) => p.k)
      : termAdjectives(data, focus.term).map((adj) => ({
        spec: adjSpec(adjItem(adj)),
        pair: pairSpec(data, adjItem(adj), focus.term),
        k: data.pairs[adj][focus.term].n[gender],
        partnerCounts: data.adjs[adj].n[gender],
      }))
  const out = []
  for (const p of partners) {
    const count = rangeSum(p.k, i0, i1)
    if (!count) continue
    const nPartner = rangeSum(p.partnerCounts, i0, i1)
    const share = nFocus ? count / nFocus : 0
    const spark = []
    for (let i = i0; i <= i1; i++) spark.push(focusCounts[i] ? p.k[i] / focusCounts[i] : null)
    out.push({
      spec: p.spec,
      pair: p.pair,
      source: p.source,
      count,
      share,
      lift: nPartner && N ? share / (nPartner / N) : null,
      spark,
    })
  }
  return { nFocus, partners: out }
}

// Score each candidate by the log ratio of its rate in the later half of the
// range to the earlier half (pooled counts, with a small pseudo-count).
export function rankCandidates(data, candidates, metric, gender, i0, i1, minCount) {
  const mid = Math.floor((i0 + i1 + 1) / 2)
  const out = []
  for (const spec of candidates) {
    const k = seriesCounts(data, spec, metric.field, gender)
    const d = metric.den(data, spec, gender)
    if (!k || !d) continue
    let kEarly = 0, dEarly = 0, kLate = 0, dLate = 0
    for (let i = i0; i <= i1; i++) {
      if (i < mid) { kEarly += k[i]; dEarly += d[i] } else { kLate += k[i]; dLate += d[i] }
    }
    const total = kEarly + kLate
    if (total < minCount || !dEarly || !dLate) continue
    const early = (kEarly + 0.5) / (dEarly + 1)
    const late = (kLate + 0.5) / (dLate + 1)
    const spark = []
    for (let i = i0; i <= i1; i++) spark.push(d[i] ? k[i] / d[i] : null)
    out.push({ spec, total, change: Math.log2(late / early), early, late, spark })
  }
  return out
}

export function rowsToCSV(rows, metric) {
  const header = ['series', 'gender', 'decade', 'value', 'ci_low', 'ci_high', 'count', 'denominator']
  const lines = [header.join(',')]
  const quote = (s) => `"${String(s).replace(/"/g, '""')}"`
  for (const row of rows) {
    for (const p of row.points) {
      lines.push([
        quote(row.spec.label), row.gender, p.decade,
        p.value ?? '', p.lo ?? '', p.hi ?? '', p.k, p.d ?? '',
      ].join(','))
    }
  }
  return `# metric: ${metric.axis}\n` + lines.join('\n') + '\n'
}
