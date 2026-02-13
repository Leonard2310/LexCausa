import React, { useMemo, useRef, useState, useEffect, useCallback } from 'react';

/* ================================================================
   ASPIC+ Metagraph — SVG visualisation of cross-attack results.

   Layout:
     Left column   → PRO  chain links  (green nodes)
     Right column  → CONTRA chain links (red nodes)
     Curved arrows → attacks (only active ones with attack_value > 0)
     Bottom        → summary bar with net_plausibility

   The component is fully self-contained — no extra dependencies.
   ================================================================ */

// ---- colour palette ----
const COLORS = {
  proBg: '#d1fae5',
  proBorder: '#10b981',
  proText: '#065f46',
  contraBg: '#fee2e2',
  contraBorder: '#ef4444',
  contraText: '#991b1b',
  attackStroke: '#f59e0b',
  attackProStroke: '#059669',
  attackProText: '#065f46',
  attackContraStroke: '#dc2626',
  attackContraText: '#991b1b',
  attackCappedStroke: '#9ca3af',
  attackCappedText: '#6b7280',
  attackText: '#92400e',
  edgePro: '#10b981',
  edgeContra: '#ef4444',
  chainArrow: '#94a3b8',
  neutralBg: '#f3f4f6',
  neutralBorder: '#d1d5db',
};

// ---- layout constants ----
const NODE_W = 260;
const NODE_H = 110;
const V_GAP = 32;
const COL_GAP = 340;
const PAD_X = 40;
const PAD_Y = 50;
const CHAIN_ARROW_GAP = 8;

// ---- small helpers ----
const pct = (v) => `${((v ?? 0) * 100).toFixed(0)}%`;
const f2 = (v) => (v ?? 0).toFixed(2);

function nodeX(col) {
  return PAD_X + col * (NODE_W + COL_GAP);
}
function nodeY(row) {
  return PAD_Y + row * (NODE_H + V_GAP);
}

// ---- CHAIN NODE ----
function ChainNode({ link, x, y, isPro, onClick, isSelected }) {
  const border = isPro ? COLORS.proBorder : COLORS.contraBorder;
  const bg = isPro ? COLORS.proBg : COLORS.contraBg;
  const text = isPro ? COLORS.proText : COLORS.contraText;
  const attacks = link.attacks_sum ?? 0;
  const nesso = link.nesso_plausibility ?? 0;

  return (
    <g
      style={{ cursor: 'pointer' }}
      onClick={() => onClick?.(link)}
    >
      <rect
        x={x}
        y={y}
        width={NODE_W}
        height={NODE_H}
        rx={10}
        ry={10}
        fill={isSelected ? '#fffbeb' : bg}
        stroke={isSelected ? '#f59e0b' : border}
        strokeWidth={isSelected ? 2.5 : 1.5}
      />
      {/* link id label */}
      <text
        x={x + NODE_W / 2}
        y={y + 20}
        textAnchor="middle"
        fill={text}
        fontWeight="700"
        fontSize="13"
      >
        {link.link_id}
      </text>
      {/* metrics row 1 */}
      <text x={x + 10} y={y + 42} fill={text} fontSize="11">
        base {f2(link.base_score)}
      </text>
      <text x={x + NODE_W / 2 + 10} y={y + 42} fill={text} fontSize="11">
        Δprec {link.precedent_delta >= 0 ? '+' : ''}{f2(link.precedent_delta)}
      </text>
      {/* metrics row 2 */}
      <text x={x + 10} y={y + 60} fill={text} fontSize="11">
        cog {f2(link.cogency)}
      </text>
      <text x={x + 90} y={y + 60} fill={text} fontSize="11">
        norm {f2(link.norm_support)}
      </text>
      <text x={x + 175} y={y + 60} fill={text} fontSize="11">
        sem {f2(link.semantics)}
      </text>
      {/* metrics row 3 — attacks + nesso */}
      <text x={x + 10} y={y + 80} fill={COLORS.attackText} fontSize="11" fontWeight="600">
        Σattack {f2(attacks)}
      </text>
      <text
        x={x + NODE_W - 10}
        y={y + 80}
        textAnchor="end"
        fill={text}
        fontSize="13"
        fontWeight="700"
      >
        nesso {f2(nesso)}
      </text>
      {/* severity / libro tag */}
      {(link.severity_category || link.libro) && (
        <text x={x + 10} y={y + 100} fill="#6b7280" fontSize="9" fontStyle="italic">
          {link.severity_category || ''}{link.severity_category && link.libro ? ' · ' : ''}{link.libro || ''}
        </text>
      )}
    </g>
  );
}

// ---- curved attack arrow ----
function AttackArrow({ x1, y1, x2, y2, value, overlap, leftToRight, isCapped }) {
  // Build a quadratic Bezier that curves outward
  const midY = (y1 + y2) / 2;
  const curveOffset = leftToRight ? -60 : 60;
  const cpx = (x1 + x2) / 2 + curveOffset;
  const cpy = midY;
  const d = `M ${x1} ${y1} Q ${cpx} ${cpy} ${x2} ${y2}`;

  // label position at the peak of the curve
  const labelX = cpx + (leftToRight ? -10 : 10);
  const labelY = cpy;

  const strokeColor = isCapped
    ? COLORS.attackCappedStroke
    : leftToRight
      ? COLORS.attackProStroke
      : COLORS.attackContraStroke;
  const textColor = isCapped
    ? COLORS.attackCappedText
    : leftToRight
      ? COLORS.attackProText
      : COLORS.attackContraText;
  const markerId = isCapped
    ? 'url(#arrowAttackCapped)'
    : leftToRight
      ? 'url(#arrowAttackPro)'
      : 'url(#arrowAttackContra)';

  return (
    <g>
      <path
        d={d}
        fill="none"
        stroke={strokeColor}
        strokeWidth={isCapped ? 1 : 1.5 + Math.min(value * 4, 3)}
        strokeDasharray={isCapped ? '4 4' : '6 3'}
        markerEnd={markerId}
        opacity={isCapped ? 0.5 : 0.85}
      />
      <rect
        x={labelX - 30}
        y={labelY - 10}
        width={60}
        height={20}
        rx={4}
        fill="white"
        stroke={strokeColor}
        strokeWidth={0.8}
        opacity={isCapped ? 0.7 : 0.95}
      />
      <text
        x={labelX}
        y={labelY + 4}
        textAnchor="middle"
        fontSize="10"
        fill={textColor}
        fontWeight="600"
      >
        {f2(value)} ({pct(overlap)})
      </text>
    </g>
  );
}

// ---- chain arrow (vertical step connection) ----
function ChainArrow({ x, y1, y2 }) {
  const startY = y1 + NODE_H + CHAIN_ARROW_GAP;
  const endY = y2 - CHAIN_ARROW_GAP;
  return (
    <line
      x1={x}
      y1={startY}
      x2={x}
      y2={endY}
      stroke={COLORS.chainArrow}
      strokeWidth={1.5}
      markerEnd="url(#arrowChain)"
    />
  );
}

// ---- Attack detail panel ----
function AttackDetailPanel({ link, onClose }) {
  if (!link) return null;
  const activeAttacks = (link.attacks_received || []).filter(
    (a) => a.attack_value >= 0.01 && a.filter_stage !== 'top_k',
  );
  const cappedAttacks = (link.attacks_received || []).filter(
    (a) => a.attack_value >= 0.01 && a.filter_stage === 'top_k',
  );
  const attacks = [...activeAttacks, ...cappedAttacks];
  return (
    <div className="metagraph-detail-panel">
      <div className="metagraph-detail-header">
        <strong>
          <span className={`role-tag ${link._isPro ? 'role-pro' : 'role-contra'}`}>
            {link._isPro ? 'PRO' : 'CONTRA'}
          </span>
          {link.link_id}
        </strong>
        <button className="metagraph-detail-close" onClick={onClose}>
          ✕
        </button>
      </div>
      <div className="metagraph-detail-body">
        <div className="metagraph-detail-row">
          <span>Base score</span>
          <strong>{f2(link.base_score)}</strong>
        </div>
        <div className="metagraph-detail-row">
          <span>Cogency</span>
          <strong>{f2(link.cogency)}</strong>
        </div>
        <div className="metagraph-detail-row">
          <span>NormSupport</span>
          <strong>{f2(link.norm_support)}</strong>
        </div>
        <div className="metagraph-detail-row">
          <span>Semantics</span>
          <strong>{f2(link.semantics)}</strong>
        </div>
        <div className="metagraph-detail-row">
          <span>Σ Attacks</span>
          <strong style={{ color: COLORS.attackText }}>
            {f2(link.attacks_sum)}
          </strong>
        </div>
        <div className="metagraph-detail-row">
          <span>Δ Precedenti</span>
          <strong>
            {link.precedent_delta >= 0 ? '+' : ''}
            {f2(link.precedent_delta)}
          </strong>
        </div>
        <div className="metagraph-detail-row metagraph-detail-total">
          <span>Nesso plausibility</span>
          <strong>{f2(link.nesso_plausibility)}</strong>
        </div>
        {attacks.length > 0 && (
          <>
            <h6 className="metagraph-detail-sub">
              Attacchi ricevuti ({attacks.length})
            </h6>
            {attacks.map((atk, i) => {
              const boosted = atk.boosted_attack ?? (atk.overlap * atk.attacker_base_score * (atk.type_multiplier || 1));
              const excess = atk.excess ?? Math.max(0, boosted - (atk.target_base_score ?? 0));
              const df = atk.damage_factor ?? 0.5;
              const isCapped = atk.filter_stage === 'top_k';
              return (
                <div key={i} className={`metagraph-attack-detail${isCapped ? ' metagraph-attack-capped' : ''}`}>
                  <span>
                    ← <span className={`role-tag ${atk.attacker_role === 'contra' ? 'role-contra' : 'role-pro'}`}>
                      {atk.attacker_role === 'contra' ? 'C' : 'P'}
                    </span>
                    {atk.attacker_link_id}
                    {isCapped && <span className="capped-badge">capped</span>}
                  </span>
                  <span className="metagraph-attack-formula">
                    boosted = {pct(atk.overlap)} × {f2(atk.attacker_base_score)}
                    {atk.type_multiplier ? ` × ${f2(atk.type_multiplier)}` : ''}
                    {' '}= {f2(boosted)}
                  </span>
                  <span className="metagraph-attack-formula">
                    excess = max(0, {f2(boosted)} − {f2(atk.target_base_score ?? 0)}) = {f2(excess)}
                  </span>
                  <span className="metagraph-attack-formula">
                    val = {f2(excess)} × {f2(df)} = <strong>{f2(atk.attack_value)}</strong>
                  </span>
                  <span className="metagraph-attack-reason">{atk.reason}</span>
                </div>
              );
            })}
          </>
        )}
        {link.precedent_influences?.length > 0 && (
          <>
            <h6 className="metagraph-detail-sub">
              Precedenti ({link.precedent_influences.length})
            </h6>
            {link.precedent_influences.map((p, i) => (
              <div key={i} className="metagraph-attack-detail">
                <span>{p.precedent_id}</span>
                <span>Δ {p.delta >= 0 ? '+' : ''}{f2(p.delta)}</span>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ---- MAIN COMPONENT ----
export default function AspicMetagraph({ aqaReport }) {
  const svgRef = useRef(null);
  const [selectedLink, setSelectedLink] = useState(null);
  const [zoom, setZoom] = useState(0.85);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const proLinks = aqaReport?.links?.pro ?? [];
  const contraLinks = aqaReport?.links?.contra ?? [];
  const chainScores = aqaReport?.chain_scores ?? {};
  const netPlaus = aqaReport?.net_plausibility ?? {};

  // Build attack edges (only active, cross-chain ones)
  const attackEdges = useMemo(() => {
    const edges = [];

    // Index by composite key "role:link_id" so pro S1->S2 ≠ contra S1->S2
    const linkIndex = {};
    proLinks.forEach((l, i) => {
      linkIndex[`support:${l.link_id}`] = { link: l, isPro: true, row: i };
    });
    contraLinks.forEach((l, i) => {
      linkIndex[`contra:${l.link_id}`] = { link: l, isPro: false, row: i };
    });

    const allLinks = [
      ...proLinks.map((l) => ({ ...l, _isPro: true })),
      ...contraLinks.map((l) => ({ ...l, _isPro: false })),
    ];

    allLinks.forEach((target) => {
      const targetIsPro = target._isPro;
      const targetCol = targetIsPro ? 0 : 1;
      const targetRow = targetIsPro
        ? proLinks.findIndex((l) => l.link_id === target.link_id)
        : contraLinks.findIndex((l) => l.link_id === target.link_id);

      (target.attacks_received || []).forEach((atk) => {
        if (!atk.attack_value || atk.attack_value < 0.01) return;

        // Attacker comes from the opposite chain
        const attackerRole = atk.attacker_role || (targetIsPro ? 'contra' : 'support');
        const attackerKey = `${attackerRole}:${atk.attacker_link_id}`;
        const attackerInfo = linkIndex[attackerKey];
        if (!attackerInfo) return;

        const isCapped = atk.filter_stage === 'top_k';

        edges.push({
          fromRow: attackerInfo.row,
          fromCol: attackerInfo.isPro ? 0 : 1,
          toRow: targetRow,
          toCol: targetCol,
          value: atk.attack_value,
          overlap: atk.overlap,
          leftToRight: attackerInfo.isPro,
          isCapped,
        });
      });
    });
    return edges;
  }, [proLinks, contraLinks]);

  // SVG dimensions
  const maxRows = Math.max(proLinks.length, contraLinks.length, 1);
  const svgW = PAD_X * 2 + NODE_W * 2 + COL_GAP;
  const svgH = PAD_Y * 2 + maxRows * (NODE_H + V_GAP) + 80;

  // Zoom handler
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.4, Math.min(2, z - e.deltaY * 0.001)));
  }, []);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (svgEl) {
      svgEl.addEventListener('wheel', handleWheel, { passive: false });
      return () => svgEl.removeEventListener('wheel', handleWheel);
    }
  }, [handleWheel]);

  const handleMouseDown = (e) => {
    if (e.button === 0 && e.target.tagName === 'svg') {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };
  const handleMouseMove = (e) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
  };
  const handleMouseUp = () => setIsPanning(false);

  if (!aqaReport || !aqaReport.enabled) return null;
  if (proLinks.length === 0 && contraLinks.length === 0) return null;

  const proNet = netPlaus.pro ?? 0;
  const contraNet = netPlaus.contra ?? 0;
  const finalNet = netPlaus.final ?? 0;
  const verdict = aqaReport.verdict ?? 'uncertain';

  return (
    <div className="metagraph-wrapper">
      {/* Legend */}
      <div className="metagraph-legend">
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.proBg, border: `2px solid ${COLORS.proBorder}` }}
          />
          PRO (Reasoner)
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.contraBg, border: `2px solid ${COLORS.contraBorder}` }}
          />
          CONTRA (Counter)
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.attackProStroke, borderRadius: 0, height: 3, width: 20, alignSelf: 'center' }}
          />
          Attacco PRO
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.attackContraStroke, borderRadius: 0, height: 3, width: 20, alignSelf: 'center' }}
          />
          Attacco CONTRA
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.attackCappedStroke, borderRadius: 0, height: 3, width: 20, alignSelf: 'center', opacity: 0.5 }}
          />
          Attacco capped (top-K)
        </span>
        <span className="metagraph-zoom-info">
          Zoom: {(zoom * 100).toFixed(0)}% · Scroll per zoom, trascina per spostare
        </span>
      </div>

      {/* SVG canvas */}
      <div className="metagraph-canvas-wrap">
        <svg
          ref={svgRef}
          width="100%"
          height={Math.max(svgH * zoom + 40, 300)}
          style={{
            cursor: isPanning ? 'grabbing' : 'grab',
            userSelect: 'none',
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <defs>
            {/* PRO attack arrowhead (green) */}
            <marker
              id="arrowAttackPro"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.attackProStroke} />
            </marker>
            {/* CONTRA attack arrowhead (red) */}
            <marker
              id="arrowAttackContra"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.attackContraStroke} />
            </marker>
            {/* Capped attack arrowhead (grey) */}
            <marker
              id="arrowAttackCapped"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.attackCappedStroke} />
            </marker>
            {/* Chain arrowhead */}
            <marker
              id="arrowChain"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.chainArrow} />
            </marker>
          </defs>

          <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
            {/* Column headers */}
            <text
              x={nodeX(0) + NODE_W / 2}
              y={PAD_Y - 16}
              textAnchor="middle"
              fontWeight="700"
              fontSize="14"
              fill={COLORS.proText}
            >
              PRO (Reasoner) — net {f2(proNet)}
            </text>
            <text
              x={nodeX(1) + NODE_W / 2}
              y={PAD_Y - 16}
              textAnchor="middle"
              fontWeight="700"
              fontSize="14"
              fill={COLORS.contraText}
            >
              CONTRA (Counter) — net {f2(contraNet)}
            </text>

            {/* PRO chain arrows */}
            {proLinks.map((_, i) =>
              i < proLinks.length - 1 ? (
                <ChainArrow
                  key={`pro-chain-${i}`}
                  x={nodeX(0) + NODE_W / 2}
                  y1={nodeY(i)}
                  y2={nodeY(i + 1)}
                />
              ) : null,
            )}

            {/* CONTRA chain arrows */}
            {contraLinks.map((_, i) =>
              i < contraLinks.length - 1 ? (
                <ChainArrow
                  key={`contra-chain-${i}`}
                  x={nodeX(1) + NODE_W / 2}
                  y1={nodeY(i)}
                  y2={nodeY(i + 1)}
                />
              ) : null,
            )}

            {/* Attack edges */}
            {attackEdges.map((edge, i) => {
              const x1 =
                edge.fromCol === 0
                  ? nodeX(0) + NODE_W
                  : nodeX(1);
              const y1 = nodeY(edge.fromRow) + NODE_H / 2;
              const x2 =
                edge.toCol === 0
                  ? nodeX(0) + NODE_W
                  : nodeX(1);
              const y2 = nodeY(edge.toRow) + NODE_H / 2;
              return (
                <AttackArrow
                  key={`attack-${i}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  value={edge.value}
                  overlap={edge.overlap}
                  leftToRight={edge.leftToRight}
                  isCapped={edge.isCapped}
                />
              );
            })}

            {/* PRO nodes */}
            {proLinks.map((link, i) => (
              <ChainNode
                key={`pro-${i}`}
                link={link}
                x={nodeX(0)}
                y={nodeY(i)}
                isPro
                onClick={(l) => setSelectedLink({ ...l, _isPro: true })}
                isSelected={selectedLink?.link_id === link.link_id && selectedLink?._isPro === true}
              />
            ))}

            {/* CONTRA nodes */}
            {contraLinks.map((link, i) => (
              <ChainNode
                key={`contra-${i}`}
                link={link}
                x={nodeX(1)}
                y={nodeY(i)}
                isPro={false}
                onClick={(l) => setSelectedLink({ ...l, _isPro: false })}
                isSelected={selectedLink?.link_id === link.link_id && selectedLink?._isPro === false}
              />
            ))}

            {/* Bottom summary */}
            <text
              x={svgW / 2}
              y={nodeY(maxRows) + 10}
              textAnchor="middle"
              fontWeight="700"
              fontSize="15"
              fill="#111827"
            >
              Plausibility finale = {f2(proNet)} − {f2(contraNet)} = {finalNet >= 0 ? '+' : ''}{f2(finalNet)}
              {'  →  '}
              {verdict === 'plausible'
                ? '✅ Plausibile'
                : verdict === 'implausible'
                  ? '❌ Implausibile'
                  : '⚠️ Incerto'}
            </text>
          </g>
        </svg>
      </div>

      {/* Detail panel (click on a node) */}
      {selectedLink && (
        <AttackDetailPanel
          link={selectedLink}
          onClose={() => setSelectedLink(null)}
        />
      )}
    </div>
  );
}
