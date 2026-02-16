import { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { getDocumentSimilarity } from '../api';
import StatusMessage from './StatusMessage';
import './SimilarityMap.css';

const TYPE_COLORS = {
  pdf: '#ef4444',
  text: '#3b82f6',
  web: '#22c55e',
};

function getColor(sourceType) {
  return TYPE_COLORS[sourceType] || '#a3a3a3';
}

export default function SimilarityMap() {
  const [data, setData] = useState(null);
  const [threshold, setThreshold] = useState(0.3);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);
  const simulationRef = useRef(null);

  // Fetch all data once on mount (threshold=0 to get everything)
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getDocumentSimilarity(0)
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
  }, []);

  // Filter edges locally by threshold
  const filteredEdges = data
    ? data.edges.filter((e) => e.similarity >= threshold)
    : [];

  const showTooltip = useCallback((event, d) => {
    const tooltip = tooltipRef.current;
    if (!tooltip || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const connCount = filteredEdges.filter(
      (e) => e.source === d.document_id || e.target === d.document_id ||
             e.source.document_id === d.document_id || e.target.document_id === d.document_id
    ).length;
    tooltip.innerHTML = `
      <div class="tt-filename">${d.filename}</div>
      <div class="tt-detail">Type: ${d.source_type}</div>
      <div class="tt-detail">Chunks: ${d.chunk_count}</div>
      <div class="tt-detail">Connections: ${connCount}</div>
    `;
    tooltip.style.left = `${event.clientX - rect.left + 12}px`;
    tooltip.style.top = `${event.clientY - rect.top - 10}px`;
    tooltip.classList.add('visible');
  }, [filteredEdges]);

  const hideTooltip = useCallback(() => {
    if (tooltipRef.current) tooltipRef.current.classList.remove('visible');
  }, []);

  // Render D3 graph
  useEffect(() => {
    if (!data || !svgRef.current) return;
    if (data.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth || 800;
    const height = svgRef.current.clientHeight || 500;

    svg.selectAll('*').remove();

    const g = svg.append('g');

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.3, 4])
      .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);

    // Build node/link data (deep copy to avoid D3 mutation issues)
    const nodes = data.nodes.map((n) => ({ ...n, id: n.document_id }));
    const links = filteredEdges
      .map((e) => ({
        source: typeof e.source === 'object' ? e.source.document_id : e.source,
        target: typeof e.target === 'object' ? e.target.document_id : e.target,
        similarity: e.similarity,
      }))
      .filter((l) => {
        const hasSource = nodes.some((n) => n.id === l.source);
        const hasTarget = nodes.some((n) => n.id === l.target);
        return hasSource && hasTarget;
      });

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d) => d.id)
        .distance((d) => 200 * (1 - d.similarity)))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d) => Math.sqrt(d.chunk_count) * 5 + 10));

    simulationRef.current = simulation;

    // Links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', (d) => 0.2 + d.similarity * 0.6)
      .attr('stroke-width', (d) => 1 + d.similarity * 3);

    // Nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer');

    node.append('circle')
      .attr('r', (d) => Math.sqrt(d.chunk_count) * 5 + 6)
      .attr('fill', (d) => getColor(d.source_type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    node.append('text')
      .text((d) => d.filename.length > 20 ? d.filename.slice(0, 18) + '...' : d.filename)
      .attr('dy', (d) => Math.sqrt(d.chunk_count) * 5 + 18)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', 'var(--color-text-secondary)');

    // Drag behavior
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    node.call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended));

    // Hover
    node.on('mouseover', (event, d) => showTooltip(event, d))
      .on('mousemove', (event, d) => showTooltip(event, d))
      .on('mouseout', hideTooltip);

    // Click to highlight
    node.on('click', (event, d) => {
      event.stopPropagation();
      setSelectedNode((prev) => prev === d.id ? null : d.id);
    });

    // Tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);
      node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [data, filteredEdges, showTooltip, hideTooltip]);

  // Handle click highlight dimming
  useEffect(() => {
    if (!svgRef.current || !data) return;
    const svg = d3.select(svgRef.current);

    if (!selectedNode) {
      svg.selectAll('.nodes g').style('opacity', 1);
      svg.selectAll('.links line').style('opacity', null);
      return;
    }

    const connectedIds = new Set([selectedNode]);
    filteredEdges.forEach((e) => {
      const src = typeof e.source === 'object' ? e.source.document_id : e.source;
      const tgt = typeof e.target === 'object' ? e.target.document_id : e.target;
      if (src === selectedNode) connectedIds.add(tgt);
      if (tgt === selectedNode) connectedIds.add(src);
    });

    svg.selectAll('.nodes g').style('opacity', (d) => connectedIds.has(d.id) ? 1 : 0.15);
    svg.selectAll('.links line').style('opacity', (d) => {
      const src = typeof d.source === 'object' ? d.source.id : d.source;
      const tgt = typeof d.target === 'object' ? d.target.id : d.target;
      return (src === selectedNode || tgt === selectedNode) ? null : 0.05;
    });
  }, [selectedNode, data, filteredEdges]);

  // Clear selection on background click
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const handler = () => setSelectedNode(null);
    svg.addEventListener('click', handler);
    return () => svg.removeEventListener('click', handler);
  }, []);

  if (loading) return <StatusMessage type="loading" message="Computing document similarity..." />;
  if (error) return <StatusMessage type="error" message={error} />;
  if (!data || data.nodes.length === 0) {
    return <div className="empty-state">No documents ingested yet. Upload some documents to see the similarity map.</div>;
  }

  return (
    <div className="similarity-map">
      <div className="similarity-controls">
        <div className="threshold-control">
          <label>Threshold</label>
          <input
            type="range"
            className="threshold-slider"
            min="0"
            max="1"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
          <span className="threshold-value">{threshold.toFixed(2)}</span>
        </div>
        <span className="edge-count">
          {filteredEdges.length} edge{filteredEdges.length !== 1 ? 's' : ''}
        </span>
        <div className="similarity-legend">
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <span key={type} className="legend-item">
              <span className="legend-dot" style={{ background: color }} />
              {type}
            </span>
          ))}
        </div>
      </div>
      <div className="graph-container" ref={containerRef}>
        <svg ref={svgRef} />
        <div className="graph-tooltip" ref={tooltipRef} />
      </div>
    </div>
  );
}
