import { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { getMetrics } from '../api';
import StatusMessage from './StatusMessage';
import './UsagePanel.css';

const EVENT_COLORS = {
  query: '#3b82f6',
  query_stream: '#3b82f6',
  embedding: '#22c55e',
  tag_generation: '#f59e0b',
  ingest: '#8b5cf6',
};

const LEGEND_ENTRIES = {
  query: '#3b82f6',
  embedding: '#22c55e',
  tag_generation: '#f59e0b',
  ingest: '#8b5cf6',
};

const TIME_RANGES = [
  { label: '5m', minutes: 5 },
  { label: '30m', minutes: 30 },
  { label: '1h', minutes: 60 },
  { label: '3h', minutes: 180 },
  { label: '12h', minutes: 720 },
  { label: '24h', minutes: 1440 },
];

function getColor(eventType) {
  return EVENT_COLORS[eventType] || '#a3a3a3';
}

export default function UsagePanel() {
  const [data, setData] = useState(null);
  const [minutes, setMinutes] = useState(60);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getMetrics(minutes)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [minutes]);

  const showTooltip = useCallback((event, d) => {
    const tooltip = tooltipRef.current;
    if (!tooltip || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    tooltip.innerHTML = `
      <div class="tt-type">${d.event_type}</div>
      <div class="tt-detail">Model: ${d.model}</div>
      <div class="tt-detail">Time: ${new Date(d.timestamp).toLocaleTimeString()}</div>
      ${d.total_tokens ? `<div class="tt-detail">Tokens: ${d.total_tokens}</div>` : ''}
      ${d.duration_ms ? `<div class="tt-detail">Duration: ${Math.round(d.duration_ms)}ms</div>` : ''}
    `;
    tooltip.style.left = `${event.clientX - rect.left + 12}px`;
    tooltip.style.top = `${event.clientY - rect.top - 10}px`;
    tooltip.classList.add('visible');
  }, []);

  const hideTooltip = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.classList.remove('visible');
  }, []);

  // Render D3 chart
  useEffect(() => {
    if (!data || !svgRef.current) return;

    const events = data.events;
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 800;
    const height = svgRef.current.clientHeight || 400;
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };

    svg.selectAll('*').remove();

    if (events.length === 0) return;

    const parseTime = (d) => new Date(d.timestamp);

    const xScale = d3.scaleTime()
      .domain(d3.extent(events, parseTime))
      .range([margin.left, width - margin.right])
      .nice();

    const maxTokens = d3.max(events, (d) => d.total_tokens || 0) || 1;
    const yScale = d3.scaleLinear()
      .domain([0, maxTokens])
      .range([height - margin.bottom, margin.top])
      .nice();

    const radiusScale = d3.scaleSqrt()
      .domain([0, maxTokens])
      .range([4, 16]);

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .selectAll('text')
      .attr('font-size', '11px');

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .attr('font-size', '11px');

    // Y-axis label
    svg.append('text')
      .attr('transform', `rotate(-90)`)
      .attr('x', -(height / 2))
      .attr('y', 16)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', 'var(--color-text-secondary)')
      .text('Tokens');

    // Data points
    svg.selectAll('circle')
      .data(events)
      .join('circle')
      .attr('cx', (d) => xScale(parseTime(d)))
      .attr('cy', (d) => yScale(d.total_tokens || 0))
      .attr('r', (d) => radiusScale(d.total_tokens || 0))
      .attr('fill', (d) => getColor(d.event_type))
      .attr('fill-opacity', 0.7)
      .attr('stroke', (d) => getColor(d.event_type))
      .attr('stroke-width', 1)
      .style('cursor', 'pointer')
      .on('mouseover', (event, d) => showTooltip(event, d))
      .on('mousemove', (event, d) => showTooltip(event, d))
      .on('mouseout', hideTooltip);
  }, [data, showTooltip, hideTooltip]);

  const totalEvents = data ? data.total : 0;
  const totalTokens = data
    ? data.events.reduce((sum, e) => sum + (e.total_tokens || 0), 0)
    : 0;
  const avgDuration = data && data.events.length > 0
    ? Math.round(
        data.events.reduce((sum, e) => sum + (e.duration_ms || 0), 0) /
        data.events.filter((e) => e.duration_ms).length || 0
      )
    : 0;

  if (loading) return <StatusMessage type="loading" message="Loading usage metrics..." />;
  if (error) return <StatusMessage type="error" message={error} />;

  return (
    <div className="usage-panel">
      <div className="usage-controls">
        <div className="time-range-buttons">
          {TIME_RANGES.map((range) => (
            <button
              key={range.minutes}
              className={minutes === range.minutes ? 'active' : ''}
              onClick={() => setMinutes(range.minutes)}
            >
              {range.label}
            </button>
          ))}
        </div>
        <div className="usage-stats">
          <div className="usage-stat">
            <span className="stat-label">Total Events</span>
            <span className="stat-value">{totalEvents}</span>
          </div>
          <div className="usage-stat">
            <span className="stat-label">Total Tokens</span>
            <span className="stat-value">{totalTokens.toLocaleString()}</span>
          </div>
          <div className="usage-stat">
            <span className="stat-label">Avg Duration</span>
            <span className="stat-value">{avgDuration}ms</span>
          </div>
        </div>
        <div className="usage-legend">
          {Object.entries(LEGEND_ENTRIES).map(([type, color]) => (
            <span key={type} className="legend-item">
              <span className="legend-dot" style={{ background: color }} />
              {type}
            </span>
          ))}
        </div>
      </div>
      <div className="usage-chart-container" ref={containerRef}>
        {data && data.events.length === 0 ? (
          <div className="empty-state">No events in this time range.</div>
        ) : (
          <svg ref={svgRef} />
        )}
        <div className="usage-tooltip" ref={tooltipRef} />
      </div>
    </div>
  );
}
