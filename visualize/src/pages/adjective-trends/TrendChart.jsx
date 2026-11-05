import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { formatValue } from './metrics.js'

const MARGIN = { top: 16, right: 130, bottom: 40, left: 64 }
// End labels need less room than the plot on narrow screens.
const NARROW_RIGHT = 72
const HEIGHT = 420
const LABEL_GAP = 14

// Rescale a row's points so its peak is 1 (for comparing shapes).
function indexToPeak(points) {
  const peak = d3.max(points, (p) => p.value)
  if (!peak) return points
  return points.map((p) => ({
    ...p,
    value: p.value == null ? null : p.value / peak,
    lo: p.lo == null ? null : p.lo / peak,
    hi: p.hi == null ? null : p.hi / peak,
  }))
}

// Push overlapping end labels apart vertically, keeping their order.
function spreadLabels(labels, minY, maxY) {
  labels.sort((a, b) => a.y - b.y)
  for (let i = 1; i < labels.length; i++) {
    labels[i].y = Math.max(labels[i].y, labels[i - 1].y + LABEL_GAP)
  }
  const overflow = labels.length ? labels[labels.length - 1].y - maxY : 0
  if (overflow > 0) {
    for (const label of labels) label.y -= overflow
    labels[0].y = Math.max(labels[0].y, minY)
    for (let i = 1; i < labels.length; i++) {
      labels[i].y = Math.max(labels[i].y, labels[i - 1].y + LABEL_GAP)
    }
  }
  return labels
}

function addTooltipRow(tip, row, point, metric) {
  const line = document.createElement('div')
  line.className = 'at-tip-row'
  const key = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  key.setAttribute('width', 16)
  key.setAttribute('height', 8)
  const stroke = document.createElementNS('http://www.w3.org/2000/svg', 'line')
  stroke.setAttribute('x1', 0)
  stroke.setAttribute('x2', 16)
  stroke.setAttribute('y1', 4)
  stroke.setAttribute('y2', 4)
  stroke.setAttribute('stroke', row.color)
  stroke.setAttribute('stroke-width', 2)
  if (row.dashed) stroke.setAttribute('stroke-dasharray', '4,3')
  key.appendChild(stroke)
  const value = document.createElement('strong')
  value.textContent = point.value == null ? '–' : formatValue(point.rawValue ?? point.value, metric)
  const label = document.createElement('span')
  label.className = 'at-tip-label'
  label.textContent = row.label
  const counts = document.createElement('span')
  counts.className = 'at-tip-counts'
  counts.textContent = point.d == null
    ? ''
    : `${point.k.toLocaleString()} / ${point.d.toLocaleString()}`
  line.append(key, value, label, counts)
  tip.appendChild(line)
}

export default function TrendChart({ rows, decades, metric, showCI, yMode, selected, onSelect }) {
  const containerRef = useRef(null)
  const tipRef = useRef(null)
  const [width, setWidth] = useState(800)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(320, entry.contentRect.width)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    d3.select(el).selectAll('svg').remove()
    if (!rows.length) return

    const right = width < 600 ? NARROW_RIGHT : MARGIN.right
    const innerW = width - MARGIN.left - right
    const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
    const series = rows.map((row) => {
      const raw = row.points.map((p) => ({ ...p, rawValue: p.value }))
      return { ...row, points: yMode === 'peak' ? indexToPeak(raw) : raw }
    })
    const isLog = yMode === 'log'
    const valid = (p) => p.value != null && (!isLog || p.value > 0)

    const x = d3.scalePoint().domain(decades).range([0, innerW]).padding(0.3)
    const extentVals = series.flatMap((s) =>
      s.points.filter(valid).flatMap((p) => (showCI && p.hi != null ? [p.value, p.hi, p.lo] : [p.value])),
    ).filter((v) => !isLog || v > 0)
    let y
    if (isLog) {
      const [lo, hi] = d3.extent(extentVals)
      y = d3.scaleLog().domain([lo ?? 0.01, hi ?? 1]).range([innerH, 0])
    } else {
      y = d3.scaleLinear().domain([0, d3.max(extentVals) || 1]).range([innerH, 0]).nice()
    }

    const svg = d3.select(el).append('svg')
      .attr('width', width).attr('height', HEIGHT)
      .attr('class', 'at-chart-svg')
    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)

    // Recessive grid and axes.
    // Log ticks are thinned to 1-2-5 steps so the grid stays recessive.
    const yTicks = isLog
      ? y.ticks().filter((v) => [1, 2, 5].includes(Math.round(v / 10 ** Math.floor(Math.log10(v) + 1e-9))))
      : y.ticks(6)
    const yFormat = isLog ? d3.format('~g') : y.tickFormat(6)
    g.append('g').attr('class', 'at-grid')
      .call(d3.axisLeft(y).tickValues(yTicks).tickSize(-innerW).tickFormat(''))
    g.append('g').attr('class', 'at-axis')
      .call(d3.axisLeft(y).tickValues(yTicks).tickFormat(yFormat).tickSizeOuter(0))
    const tickEvery = Math.ceil(decades.length / Math.max(2, Math.floor(innerW / 60)))
    g.append('g').attr('class', 'at-axis').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickSizeOuter(0)
        .tickValues(decades.filter((_, i) => i % tickEvery === 0))
        .tickFormat((d) => `${d}s`))
    g.append('text').attr('class', 'at-axis-label')
      .attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -48)
      .attr('text-anchor', 'middle')
      .text(yMode === 'peak' ? `${metric.axis} (peak = 1)` : metric.axis)

    // Confidence bands.
    if (showCI && metric.den) {
      const area = d3.area()
        .defined((p) => valid(p) && p.lo != null && (!isLog || p.lo > 0))
        .x((p) => x(p.decade)).y0((p) => y(p.lo)).y1((p) => y(p.hi))
      g.append('g').selectAll('path').data(series).join('path')
        .attr('fill', (s) => s.color).attr('fill-opacity', (s) => (s.dashed ? 0.08 : 0.14))
        .attr('d', (s) => area(s.points))
    }

    // Lines: solid through observed decades, with a faint bridge over gaps.
    const line = d3.line().defined(valid).x((p) => x(p.decade)).y((p) => y(p.value))
    const lines = g.append('g')
    lines.selectAll('path.at-bridge').data(series).join('path')
      .attr('class', 'at-bridge').attr('fill', 'none').attr('stroke', (s) => s.color)
      .attr('stroke-width', 1).attr('stroke-opacity', 0.35).attr('stroke-dasharray', '2,3')
      .attr('d', (s) => line(s.points.filter(valid)))
    lines.selectAll('path.at-line').data(series).join('path')
      .attr('class', 'at-line').attr('fill', 'none').attr('stroke', (s) => s.color)
      .attr('stroke-width', 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
      .attr('stroke-dasharray', (s) => (s.dashed ? '6,4' : null))
      .attr('d', (s) => line(s.points))

    // Markers, with the selected one ringed.
    const markers = g.append('g')
    series.forEach((s) => {
      markers.selectAll(null).data(s.points.filter(valid)).join('circle')
        .attr('cx', (p) => x(p.decade)).attr('cy', (p) => y(p.value))
        .attr('r', (p) => (selected?.key === s.spec.key && selected?.decade === p.decade ? 6 : 4))
        .attr('fill', (p) => (s.dashed && !(selected?.key === s.spec.key && selected?.decade === p.decade)
          ? 'var(--at-surface)' : s.color))
        .attr('stroke', (p) => (selected?.key === s.spec.key && selected?.decade === p.decade
          ? 'var(--text-h)' : s.dashed ? s.color : 'var(--at-surface)'))
        .attr('stroke-width', 2)
    })

    // Direct end labels when few enough lines to stay readable.
    if (series.length <= 4) {
      const labels = series.map((s) => {
        const last = [...s.points].reverse().find(valid)
        return last && { text: s.label, color: s.color, y: y(last.value), x: x(last.decade) }
      }).filter(Boolean)
      spreadLabels(labels, 0, innerH)
      const lastX = d3.max(labels, (l) => l.x) ?? innerW
      g.append('g').selectAll('text').data(labels).join('text')
        .attr('class', 'at-end-label')
        .attr('x', lastX + 10).attr('y', (l) => l.y).attr('dy', '0.32em')
        .text((l) => l.text)
      g.append('g').selectAll('line').data(labels).join('line')
        .attr('x1', (l) => l.x + 6).attr('x2', lastX + 7)
        .attr('y1', (l) => l.y).attr('y2', (l) => l.y)
        .attr('stroke', (l) => l.color).attr('stroke-width', 1.5)
    }

    // Hover layer: crosshair snaps to the nearest decade; one tooltip lists every series.
    const crosshair = g.append('line').attr('class', 'at-crosshair')
      .attr('y1', 0).attr('y2', innerH).style('display', 'none')
    const nearestDecade = (mx) => decades.reduce((best, d) =>
      Math.abs(x(d) - mx) < Math.abs(x(best) - mx) ? d : best, decades[0])
    const tip = tipRef.current

    g.append('rect').attr('class', 'at-hit')
      .attr('width', innerW).attr('height', innerH).attr('fill', 'transparent')
      .style('cursor', 'pointer')
      .on('pointermove', (event) => {
        const [mx] = d3.pointer(event)
        const decade = nearestDecade(mx)
        crosshair.attr('x1', x(decade)).attr('x2', x(decade)).style('display', null)
        tip.replaceChildren()
        const head = document.createElement('div')
        head.className = 'at-tip-head'
        head.textContent = `${decade}s — click for examples`
        tip.appendChild(head)
        series
          .map((s) => ({ s, p: s.points.find((p) => p.decade === decade) }))
          .sort((a, b) => (b.p?.rawValue ?? -1) - (a.p?.rawValue ?? -1))
          .forEach(({ s, p }) => p && addTooltipRow(tip, s, p, metric))
        tip.style.display = 'block'
        const box = tip.getBoundingClientRect()
        const left = event.clientX + 16 + box.width > window.innerWidth
          ? event.clientX - 16 - box.width : event.clientX + 16
        tip.style.left = `${left}px`
        tip.style.top = `${Math.min(event.clientY - 10, window.innerHeight - box.height - 8)}px`
      })
      .on('pointerleave', () => {
        crosshair.style('display', 'none')
        tip.style.display = 'none'
      })
      .on('click', (event) => {
        // Pick the series whose point at this decade is vertically closest.
        const [mx, my] = d3.pointer(event)
        const decade = nearestDecade(mx)
        let best = null
        for (const s of series) {
          const p = s.points.find((q) => q.decade === decade)
          if (!p || !valid(p)) continue
          const dist = Math.abs(y(p.value) - my)
          if (!best || dist < best.dist) best = { dist, s }
        }
        if (best) onSelect({ key: best.s.spec.key, decade, gender: best.s.gender })
      })
  }, [rows, decades, metric, showCI, yMode, selected, onSelect, width])

  return (
    <>
      <div ref={containerRef} className="at-chart" />
      <div ref={tipRef} className="tooltip at-tooltip" style={{ display: 'none' }} />
    </>
  )
}
