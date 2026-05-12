import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

export default function GenderClassifier() {
  const [data, setData] = useState([])
  const [decades, setDecades] = useState([])
  const [selectedDecade, setSelectedDecade] = useState(null)
  const [topN, setTopN] = useState(20)
  const [mode, setMode] = useState('decade') // 'decade' | 'term'
  const [selectedTerm, setSelectedTerm] = useState('')
  const [terms, setTerms] = useState([])
  const barRef = useRef(null)
  const lineRef = useRef(null)
  const tooltip = useRef(null)

  useEffect(() => {
    d3.tsv('/data/coefficients.tsv', (row) => ({
      decade: +row.decade,
      term: row.term,
      coefficient: +row.coefficient,
    })).then((rows) => {
      const allDecades = [...new Set(rows.map((r) => r.decade))].sort((a, b) => a - b)
      const allTerms = [...new Set(rows.map((r) => r.term))].sort()
      setData(rows)
      setDecades(allDecades)
      setSelectedDecade(allDecades[allDecades.length - 1])
      setTerms(allTerms)
      setSelectedTerm(allTerms[0])
    })
  }, [])

  // Bar chart: top terms for a given decade
  useEffect(() => {
    if (!data.length || !selectedDecade || mode !== 'decade') return
    const container = barRef.current
    if (!container) return

    const decadeData = data.filter((d) => d.decade === selectedDecade)
    const sorted = [...decadeData].sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
    const top = sorted.slice(0, topN)
    top.sort((a, b) => a.coefficient - b.coefficient)

    const margin = { top: 10, right: 120, bottom: 40, left: 120 }
    const width = container.clientWidth - margin.left - margin.right
    const height = Math.max(400, top.length * 22)

    d3.select(container).selectAll('*').remove()
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleLinear()
      .domain([d3.min(top, (d) => d.coefficient), d3.max(top, (d) => d.coefficient)])
      .nice()
      .range([0, width])

    const y = d3.scaleBand()
      .domain(top.map((d) => d.term))
      .range([0, height])
      .padding(0.2)

    svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).ticks(6))
    svg.append('g').call(d3.axisLeft(y))

    const zero = x(0)
    svg.append('line')
      .attr('x1', zero).attr('x2', zero)
      .attr('y1', 0).attr('y2', height)
      .attr('stroke', '#d1d5db').attr('stroke-dasharray', '4,2')

    svg.selectAll('rect')
      .data(top)
      .join('rect')
      .attr('x', (d) => d.coefficient >= 0 ? zero : x(d.coefficient))
      .attr('y', (d) => y(d.term))
      .attr('width', (d) => Math.abs(x(d.coefficient) - zero))
      .attr('height', y.bandwidth())
      .attr('fill', (d) => d.coefficient > 0 ? 'var(--male)' : 'var(--female)')
      .attr('opacity', 0.8)
      .on('mousemove', (event, d) => {
        const tip = tooltip.current
        if (!tip) return
        tip.style.display = 'block'
        tip.style.left = event.clientX + 12 + 'px'
        tip.style.top = event.clientY - 10 + 'px'
        tip.textContent = `${d.term}: ${d.coefficient.toFixed(3)}`
      })
      .on('mouseleave', () => { if (tooltip.current) tooltip.current.style.display = 'none' })

    svg.append('text')
      .attr('x', width / 2).attr('y', height + 34)
      .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', 'var(--text-muted)')
      .text('← more female-coded   coefficient   more male-coded →')
  }, [data, selectedDecade, topN, mode])

  // Line chart: coefficient trajectory for a term across decades
  useEffect(() => {
    if (!data.length || !selectedTerm || mode !== 'term') return
    const container = lineRef.current
    if (!container) return

    const termData = data
      .filter((d) => d.term === selectedTerm)
      .sort((a, b) => a.decade - b.decade)

    const margin = { top: 20, right: 30, bottom: 50, left: 60 }
    const width = container.clientWidth - margin.left - margin.right
    const height = 320

    d3.select(container).selectAll('*').remove()
    const svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const x = d3.scaleLinear()
      .domain(d3.extent(termData, (d) => d.decade))
      .range([0, width])

    const yExt = d3.extent(termData, (d) => d.coefficient)
    const yMax = Math.max(Math.abs(yExt[0]), Math.abs(yExt[1]))
    const y = d3.scaleLinear()
      .domain([-yMax, yMax]).nice()
      .range([height, 0])

    svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x).tickFormat((d) => `${d}s`).ticks(termData.length))
    svg.append('g').call(d3.axisLeft(y).ticks(6))

    svg.append('line')
      .attr('x1', 0).attr('x2', width)
      .attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', '#d1d5db').attr('stroke-dasharray', '4,2')

    const line = d3.line()
      .x((d) => x(d.decade))
      .y((d) => y(d.coefficient))
      .curve(d3.curveMonotoneX)

    svg.append('path')
      .datum(termData)
      .attr('fill', 'none')
      .attr('stroke', 'var(--accent)')
      .attr('stroke-width', 2)
      .attr('d', line)

    svg.selectAll('circle')
      .data(termData)
      .join('circle')
      .attr('cx', (d) => x(d.decade))
      .attr('cy', (d) => y(d.coefficient))
      .attr('r', 4)
      .attr('fill', (d) => d.coefficient >= 0 ? 'var(--male)' : 'var(--female)')
      .attr('stroke', '#fff').attr('stroke-width', 1.5)
      .on('mousemove', (event, d) => {
        const tip = tooltip.current
        if (!tip) return
        tip.style.display = 'block'
        tip.style.left = event.clientX + 12 + 'px'
        tip.style.top = event.clientY - 10 + 'px'
        tip.textContent = `${d.decade}s: ${d.coefficient.toFixed(3)}`
      })
      .on('mouseleave', () => { if (tooltip.current) tooltip.current.style.display = 'none' })

    svg.append('text')
      .attr('x', width / 2).attr('y', height + 42)
      .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', 'var(--text-muted)')
      .text('Decade')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2).attr('y', -46)
      .attr('text-anchor', 'middle').attr('font-size', 12).attr('fill', 'var(--text-muted)')
      .text('Coefficient (+ = male, − = female)')
  }, [data, selectedTerm, mode])

  return (
    <div>
      <div className="page-header">
        <h1>Gender Classifier Coefficients</h1>
        <p>
          Per-decade logistic regression coefficients for fashion terms predicting character gender.
          Positive values index male characters; negative values index female characters.
        </p>
      </div>

      <div className="controls">
        <label>
          View:
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="decade">Top terms by decade</option>
            <option value="term">Term trajectory over time</option>
          </select>
        </label>

        {mode === 'decade' && (
          <>
            <label>
              Decade:
              <select value={selectedDecade ?? ''} onChange={(e) => setSelectedDecade(+e.target.value)}>
                {decades.map((d) => (
                  <option key={d} value={d}>{d}s</option>
                ))}
              </select>
            </label>
            <label>
              Show top:
              <select value={topN} onChange={(e) => setTopN(+e.target.value)}>
                {[10, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <span style={{ display: 'inline-block', width: 10, height: 10, background: 'var(--male)', borderRadius: 2, marginRight: 4 }} />male-coded
              <span style={{ display: 'inline-block', width: 10, height: 10, background: 'var(--female)', borderRadius: 2, marginLeft: 12, marginRight: 4 }} />female-coded
            </span>
          </>
        )}

        {mode === 'term' && (
          <label>
            Term:
            <select value={selectedTerm} onChange={(e) => setSelectedTerm(e.target.value)}>
              {terms.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        )}
      </div>

      <div className="chart-container">
        {mode === 'decade' && <div ref={barRef} />}
        {mode === 'term' && <div ref={lineRef} />}
      </div>

      <div ref={tooltip} className="tooltip" style={{ display: 'none' }} />
    </div>
  )
}
