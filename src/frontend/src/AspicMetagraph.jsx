import { useMemo, useRef, useState, useEffect, useCallback } from 'react';

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
  precBg: '#ede9fe',
  precBorder: '#7c3aed',
  precText: '#4c1d95',
  precEdge: '#8b5cf6',
};

// ---- layout constants ----
const NODE_W = 260;
const NODE_H = 110;
const V_GAP = 32;
const COL_GAP = 340;
const PAD_X = 40;
const PAD_Y = 90;
const CHAIN_ARROW_GAP = 8;
const PREC_NODE_W = 200;
const PREC_NODE_H = 70;
const PREC_GAP_X = 30;
const PREC_GAP_Y = 24;
const EMPTY_LINKS = Object.freeze([]);

// ---- small helpers ----
const pct = (v) => `${((v ?? 0) * 100).toFixed(0)}%`;
const f2 = (v) => (v ?? 0).toFixed(2);

// Render Italian domain/book tags in English (falls back to the raw value).
const EN_TERMS = {
  amministrativo: 'administrative',
  civile: 'civil',
  penale: 'criminal',
  misto: 'mixed',
};
const enTerm = (v) => (v ? (EN_TERMS[String(v).toLowerCase()] ?? v) : '');

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
        norms {link.norm_support_details?.citation_count ?? '?'}
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
        link plausibility {f2(nesso)}
      </text>
      {/* severity / libro tag */}
      {(link.severity_category || link.libro) && (
        <text x={x + 10} y={y + 100} fill="#6b7280" fontSize="9" fontStyle="italic">
          {enTerm(link.severity_category)}{link.severity_category && link.libro ? ' · ' : ''}{enTerm(link.libro)}
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

// ---- PRECEDENT NODE (external, purple) ----
function PrecedentNode({ prec, x, y, delta, onClick, isSelected }) {
  const title = prec.title || prec.precedent_id || 'Precedente';
  const shortTitle = title.length > 28 ? title.slice(0, 25) + '…' : title;
  const stanceLabel = prec._stance === 1 ? 'favorevole' : prec._stance === -1 ? 'contrario' : 'neutro';
  const titleClipId = `prec-title-clip-${String(prec.precedent_id || title)
    .replace(/[^a-zA-Z0-9_-]/g, '')
    .slice(0, 24)}-${Math.round(x)}-${Math.round(y)}`;

  return (
    <g style={{ cursor: 'pointer' }} onClick={() => onClick?.(prec)}>
      <defs>
        <clipPath id={titleClipId}>
          <rect x={x + 12} y={y + 6} width={PREC_NODE_W - 20} height={16} />
        </clipPath>
      </defs>
      <rect
        x={x}
        y={y}
        width={PREC_NODE_W}
        height={PREC_NODE_H}
        rx={12}
        ry={12}
        fill={isSelected ? '#faf5ff' : COLORS.precBg}
        stroke={isSelected ? '#f59e0b' : COLORS.precBorder}
        strokeWidth={isSelected ? 2.5 : 1.5}
        strokeDasharray="4 2"
      />
      {/* title */}
      <text
        x={x + 12}
        y={y + 18}
        clipPath={`url(#${titleClipId})`}
        fill={COLORS.precText}
        fontWeight="700"
        fontSize="11"
      >
        {shortTitle}
      </text>
      {/* delta */}
      <text x={x + 10} y={y + 38} fill={COLORS.precText} fontSize="11" fontWeight="600">
        Δ {delta >= 0 ? '+' : ''}{f2(delta)}
      </text>
      {/* stance */}
      <text x={x + PREC_NODE_W - 10} y={y + 38} textAnchor="end" fill={COLORS.precText} fontSize="10">
        {stanceLabel}
      </text>
      {/* bindingness + recency */}
      {prec._bind != null && (
        <text x={x + 10} y={y + 55} fill="#6b7280" fontSize="9">
          bind {f2(prec._bind)} · sim {f2(prec._sim)} · rec {f2(prec._rec)} · conf {f2(prec._conf)}
        </text>
      )}
    </g>
  );
}

// ---- PRECEDENT EDGE (dashed purple arrow) ----
function PrecedentEdge({ x1, y1, x2, y2 }) {
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const cpX = midX + (x1 < x2 ? -30 : 30);
  const cpY = midY - 20;
  const d = `M ${x1} ${y1} Q ${cpX} ${cpY} ${x2} ${y2}`;
  return (
    <path
      d={d}
      fill="none"
      stroke={COLORS.precEdge}
      strokeWidth={1.5}
      strokeDasharray="6 3"
      markerEnd="url(#arrowPrec)"
      opacity={0.8}
    />
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
          <span>Δ Precedents</span>
          <strong>
            {link.precedent_delta >= 0 ? '+' : ''}
            {f2(link.precedent_delta)}
          </strong>
        </div>
        <div className="metagraph-detail-row metagraph-detail-total">
          <span>Link plausibility</span>
          <strong>{f2(link.nesso_plausibility)}</strong>
        </div>
        {attacks.length > 0 && (
          <>
            <h6 className="metagraph-detail-sub">
              Received attacks ({attacks.length})
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
export default function AspicMetagraph({ aqaReport, reasonerIr, counterIr, forPdf = false }) {
  const svgRef = useRef(null);
  const canvasWrapRef = useRef(null);
  const [selectedLink, setSelectedLink] = useState(null);
  const [selectedPrec, setSelectedPrec] = useState(null);
  const [zoom, setZoom] = useState(0.85);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [hasManualViewport, setHasManualViewport] = useState(false);

  const proLinksRaw = aqaReport?.links?.pro;
  const contraLinksRaw = aqaReport?.links?.contra;
  const proLinks = useMemo(
    () => (Array.isArray(proLinksRaw) ? proLinksRaw : EMPTY_LINKS),
    [proLinksRaw],
  );
  const contraLinks = useMemo(
    () => (Array.isArray(contraLinksRaw) ? contraLinksRaw : EMPTY_LINKS),
    [contraLinksRaw],
  );
  const netPlaus = aqaReport?.net_plausibility ?? {};

  // Build precedent nodes with positions and connections
  const { precNodes, precEdges } = useMemo(() => {
    const nodes = [];
    const edges = [];
    const seen = new Set();

    const processIr = (ir, aLinks, isPro) => {
      if (!ir) return;
      const pNodes = ir.precedent_nodes || [];
      const pLinks = ir.precedent_links || [];
      if (pNodes.length === 0) return;

      // Build a map from precedent node ID → metadata
      const pMeta = {};
      pNodes.forEach((n) => { pMeta[n.id] = n; });

      // Collect delta info from AQA links
      const deltaByPrecId = {};
      const influenceByPrecId = {};
      const usedPrecedentIds = new Set();
      aLinks.forEach((link) => {
        (link.precedent_influences || []).forEach((inf) => {
          const pid = String(inf.precedent_id ?? '').trim();
          if (pid) {
            usedPrecedentIds.add(pid);
            deltaByPrecId[pid] = (deltaByPrecId[pid] || 0) + (inf.delta || 0);
            if (!influenceByPrecId[pid]) influenceByPrecId[pid] = inf;
          }
        });
      });

      // For each precedent node, find which step IDs it links to
      const precTargets = {};
      pLinks.forEach((e) => {
        const from = String(e.from ?? '').trim();
        if (!from) return;
        if (!precTargets[from]) precTargets[from] = [];
        precTargets[from].push(e.to);
      });

      // Build the index of link_id → row for the chain
      const chainIndex = {};
      aLinks.forEach((l, i) => { chainIndex[l.conclusion_id] = i; });
      // Also index by premise step ids
      aLinks.forEach((l, i) => {
        (l.premise_ids || []).forEach((pid) => { chainIndex[pid] = i; });
      });

      // Create a node entry for each unique precedent
      pNodes.forEach((pn, pidx) => {
        const precId = String(pn.precedent_id || pn.id || '').trim();
        if (!precId || !usedPrecedentIds.has(precId)) return;

        const inf = influenceByPrecId[precId] || {};
        let stance = inf.stance ?? 0;
        if (typeof stance === 'string') {
          const s = stance.trim().toLowerCase();
          if (s === 'support' || s === 'pro' || s === 'favour') stance = 1;
          else if (s === 'neutral') stance = 0;
          else if (s === 'against' || s === 'contra' || s === 'contradict') stance = -1;
          else stance = 0;
        }
        if (typeof stance !== 'number' || Number.isNaN(stance)) stance = 0;
        if (stance < 0) return;
        stance = stance > 0 ? 1 : 0;

        const dedup = `${isPro ? 'pro' : 'contra'}:${precId}`;
        if (seen.has(dedup)) return;
        const totalDelta = deltaByPrecId[precId] || 0;
        const targets = [
          ...(precTargets[String(pn.id ?? '').trim()] || []),
          ...(precTargets[precId] || []),
        ];

        // Find which chain rows this precedent connects to
        const targetRows = [];
        targets.forEach((t) => {
          // t can be S1, S2, A1.P1, etc.
          if (chainIndex[t] != null) {
            targetRows.push(chainIndex[t]);
          } else {
            // Try matching step id in link_id like "S1->S2" (premises)
            aLinks.forEach((l, i) => {
              if (l.link_id && l.link_id.includes(t)) {
                targetRows.push(i);
              }
            });
          }
        });
        const uniqueRows = [...new Set(targetRows)];
        if (uniqueRows.length === 0) return;

        seen.add(dedup);

        // Position: outside the chain column
        const col = isPro ? 0 : 1;
        const anchorRow = uniqueRows.length > 0 ? Math.min(...uniqueRows) : pidx;
        const px = isPro
          ? nodeX(0) - PREC_NODE_W - PREC_GAP_X
          : nodeX(1) + NODE_W + PREC_GAP_X;
        const py = nodeY(anchorRow) + pidx * (PREC_NODE_H + PREC_GAP_Y);

        const node = {
          ...pn,
          _isPro: isPro,
          _x: px,
          _y: py,
          _delta: totalDelta,
          _stance: stance,
          _bind: inf.bindingness,
          _sim: inf.similarity,
          _rec: inf.recency,
          _conf: inf.confidence,
          _targetRows: uniqueRows,
          _col: col,
        };
        nodes.push(node);
      });
    };

    processIr(reasonerIr, proLinks, true);
    processIr(counterIr, contraLinks, false);

    // Avoid vertical overlaps for precedent nodes per side.
    const resolveSideOverlaps = (isProSide) => {
      const sideNodes = nodes
        .filter((n) => n._isPro === isProSide)
        .sort((a, b) => a._y - b._y);

      let nextFreeY = PAD_Y;
      const minGap = 8;
      sideNodes.forEach((n) => {
        if (n._y < nextFreeY) {
          n._y = nextFreeY;
        }
        nextFreeY = n._y + PREC_NODE_H + minGap;
      });
    };

    resolveSideOverlaps(true);
    resolveSideOverlaps(false);

    // Create edges to chain nodes after positions are finalized.
    nodes.forEach((n) => {
      (n._targetRows || []).forEach((row) => {
        edges.push({
          precX: n._x + (n._isPro ? PREC_NODE_W : 0),
          precY: n._y + PREC_NODE_H / 2,
          chainX: n._isPro ? nodeX(0) : nodeX(1) + NODE_W,
          chainY: nodeY(row) + NODE_H / 2,
          isPro: n._isPro,
        });
      });
    });

    return { precNodes: nodes, precEdges: edges };
  }, [reasonerIr, counterIr, proLinks, contraLinks]);

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

  // Graph bounds + viewport dimensions (robust centering/fit on all layouts)
  const {
    svgW,
    svgH,
    baseOffsetX,
    baseOffsetY,
    summaryY,
    summaryCenterX,
  } = useMemo(() => {
    const proLeft = nodeX(0);
    const proRight = nodeX(0) + NODE_W;
    const contraLeft = nodeX(1);
    const contraRight = nodeX(1) + NODE_W;

    const proBottom = proLinks.length > 0
      ? nodeY(proLinks.length - 1) + NODE_H
      : PAD_Y + NODE_H;
    const contraBottom = contraLinks.length > 0
      ? nodeY(contraLinks.length - 1) + NODE_H
      : PAD_Y + NODE_H;
    const chainBottom = Math.max(proBottom, contraBottom);

    const precMinX = precNodes.length > 0
      ? Math.min(...precNodes.map((p) => p._x))
      : Math.min(proLeft, contraLeft);
    const precMaxX = precNodes.length > 0
      ? Math.max(...precNodes.map((p) => p._x + PREC_NODE_W))
      : Math.max(proRight, contraRight);
    const precMinY = precNodes.length > 0
      ? Math.min(...precNodes.map((p) => p._y))
      : PAD_Y;
    const precMaxY = precNodes.length > 0
      ? Math.max(...precNodes.map((p) => p._y + PREC_NODE_H))
      : chainBottom;

    const graphMinX = Math.min(proLeft, contraLeft, precMinX);
    const graphMaxX = Math.max(proRight, contraRight, precMaxX);
    const headerTop = PAD_Y - 32;
    const nextSummaryY = Math.max(chainBottom, precMaxY) + 52;
    const graphMinY = Math.min(headerTop, precMinY);
    const graphMaxY = Math.max(chainBottom, precMaxY, nextSummaryY + 24);

    const viewPadX = 36;
    const viewPadY = 28;
    return {
      svgW: Math.ceil(graphMaxX - graphMinX + viewPadX * 2),
      svgH: Math.ceil(graphMaxY - graphMinY + viewPadY * 2),
      baseOffsetX: -graphMinX + viewPadX,
      baseOffsetY: -graphMinY + viewPadY,
      summaryY: nextSummaryY,
      summaryCenterX: (graphMinX + graphMaxX) / 2,
    };
  }, [proLinks, contraLinks, precNodes]);

  const viewSignature = useMemo(
    () => [
      proLinks.map((l) => l.link_id).join('|'),
      contraLinks.map((l) => l.link_id).join('|'),
      precNodes.map((p) => String(p.precedent_id || p.id || '')).join('|'),
    ].join('::'),
    [proLinks, contraLinks, precNodes],
  );

  // Zoom handler
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setHasManualViewport(true);
    setZoom((z) => Math.max(0.4, Math.min(2, z - e.deltaY * 0.001)));
  }, []);

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      setHasManualViewport(false);
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [viewSignature]);

  useEffect(() => {
    if (hasManualViewport) return;
    const wrapEl = canvasWrapRef.current;
    if (!wrapEl) return;

    const viewportW = Math.max(320, wrapEl.clientWidth || svgW);
    const viewportH = Math.max(300, wrapEl.clientHeight || 500);
    const fitX = (viewportW - 24) / svgW;
    const fitY = (viewportH - 24) / svgH;
    const nextZoom = Math.max(0.45, Math.min(1, fitX, fitY));

    const scaledW = svgW * nextZoom;
    const scaledH = svgH * nextZoom;
    const centeredPanX = scaledW < viewportW ? (viewportW - scaledW) / 2 : 0;
    const centeredPanY = scaledH < viewportH ? (viewportH - scaledH) / 2 : 0;

    const frameId = window.requestAnimationFrame(() => {
      setZoom(nextZoom);
      setPan({ x: centeredPanX, y: centeredPanY });
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [hasManualViewport, svgW, svgH]);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (svgEl) {
      svgEl.addEventListener('wheel', handleWheel, { passive: false });
      return () => svgEl.removeEventListener('wheel', handleWheel);
    }
  }, [handleWheel]);

  const handleMouseDown = (e) => {
    if (e.button === 0 && e.target.tagName === 'svg') {
      setHasManualViewport(true);
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
          Primary Thesis (Reasoner)
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.contraBg, border: `2px solid ${COLORS.contraBorder}` }}
          />
          Counter-thesis (Counter)
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.attackProStroke, borderRadius: 0, height: 3, width: 20, alignSelf: 'center' }}
          />
          Primary Thesis Attack
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.attackContraStroke, borderRadius: 0, height: 3, width: 20, alignSelf: 'center' }}
          />
          Counter-thesis Attack
        </span>
        <span className="metagraph-legend-item">
          <span
            className="metagraph-legend-swatch"
            style={{ background: COLORS.attackCappedStroke, borderRadius: 0, height: 3, width: 20, alignSelf: 'center', opacity: 0.5 }}
          />
          Capped Attack (top-K)
        </span>
        {precNodes.length > 0 && (
          <span className="metagraph-legend-item">
            <span
              className="metagraph-legend-swatch"
              style={{ background: COLORS.precBg, border: `2px solid ${COLORS.precBorder}` }}
            />
            Precedente (Δ)
          </span>
        )}
        <span className="metagraph-zoom-info">
          Zoom: {(zoom * 100).toFixed(0)}% · Scroll to zoom, drag to move
        </span>
      </div>

      {/* SVG canvas */}
      <div className="metagraph-canvas-wrap" ref={canvasWrapRef}>
        <svg
          ref={svgRef}
          width="100%"
          height={forPdf ? 'auto' : Math.max(svgH * zoom + 40, 300)}
          viewBox={forPdf ? `0 0 ${Math.ceil(svgW * zoom)} ${Math.max(Math.ceil(svgH * zoom) + 40, 300)}` : undefined}
          preserveAspectRatio={forPdf ? 'xMidYMid meet' : undefined}
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
            {/* Precedent arrowhead (purple) */}
            <marker
              id="arrowPrec"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={COLORS.precEdge} />
            </marker>
          </defs>

          <g transform={`translate(${pan.x},${pan.y})`}>
            <g transform={`scale(${zoom}) translate(${baseOffsetX},${baseOffsetY})`}>
            {/* Column headers */}
            <text
              x={nodeX(0) + NODE_W / 2}
              y={PAD_Y - 16}
              textAnchor="middle"
              fontWeight="700"
              fontSize="14"
              fill={COLORS.proText}
            >
              Primary Thesis (Reasoner) — net {f2(proNet)}
            </text>
            <text
              x={nodeX(1) + NODE_W / 2}
              y={PAD_Y - 16}
              textAnchor="middle"
              fontWeight="700"
              fontSize="14"
              fill={COLORS.contraText}
            >
              Counter-thesis (Counter) — net {f2(contraNet)}
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
                onClick={(l) => { setSelectedLink({ ...l, _isPro: true }); setSelectedPrec(null); }}
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
                onClick={(l) => { setSelectedLink({ ...l, _isPro: false }); setSelectedPrec(null); }}
                isSelected={selectedLink?.link_id === link.link_id && selectedLink?._isPro === false}
              />
            ))}

            {/* Precedent edges (behind nodes) */}
            {precEdges.map((e, i) => (
              <PrecedentEdge
                key={`prec-edge-${i}`}
                x1={e.precX}
                y1={e.precY}
                x2={e.chainX}
                y2={e.chainY}
              />
            ))}

            {/* Precedent nodes */}
            {precNodes.map((prec, i) => (
              <PrecedentNode
                key={`prec-${i}`}
                prec={prec}
                x={prec._x}
                y={prec._y}
                delta={prec._delta}
                onClick={(p) => { setSelectedPrec(p); setSelectedLink(null); }}
                isSelected={selectedPrec?.precedent_id === prec.precedent_id && selectedPrec?._isPro === prec._isPro}
              />
            ))}

            {/* Bottom summary */}
            <text
              x={summaryCenterX}
              y={summaryY}
              textAnchor="middle"
              fontWeight="700"
              fontSize="15"
              fill="#111827"
            >
              Final Plausibility = {f2(proNet)} − {f2(contraNet)} = {finalNet >= 0 ? '+' : ''}{f2(finalNet)}
              {'  →  '}
              {verdict === 'plausible'
                ? 'Plausible'
                : verdict === 'implausible'
                    ? 'Implausible'
                  : 'Uncertain'}
            </text>
            </g>
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

      {/* Detail panel for precedent node */}
      {selectedPrec && (
        <div className="metagraph-detail-panel">
          <div className="metagraph-detail-header">
            <strong>
              <span className="role-tag" style={{ background: COLORS.precBg, color: COLORS.precText }}>
                PREC
              </span>
              {selectedPrec.precedent_id || selectedPrec.id}
            </strong>
            <button className="metagraph-detail-close" onClick={() => setSelectedPrec(null)}>
              ✕
            </button>
          </div>
          <div className="metagraph-detail-body">
            <div className="metagraph-detail-row">
              <span>Titolo</span>
              <strong style={{ fontSize: '0.8rem', maxWidth: '300px', textAlign: 'right' }}>{selectedPrec.title || '—'}</strong>
            </div>
            <div className="metagraph-detail-row">
              <span>Δ Delta</span>
              <strong style={{ color: selectedPrec._delta > 0 ? COLORS.proText : selectedPrec._delta < 0 ? COLORS.contraText : '#6b7280' }}>
                {selectedPrec._delta >= 0 ? '+' : ''}{f2(selectedPrec._delta)}
              </strong>
            </div>
            <div className="metagraph-detail-row">
              <span>Stance</span>
              <strong>{selectedPrec._stance === 1 ? 'Support (+1)' : selectedPrec._stance === -1 ? 'Against (−1)' : 'Neutral (0)'}</strong>
            </div>
            <div className="metagraph-detail-row">
              <span>Bindingness</span>
              <strong>{f2(selectedPrec._bind)}</strong>
            </div>
            <div className="metagraph-detail-row">
              <span>Similarity</span>
              <strong>{f2(selectedPrec._sim)}</strong>
            </div>
            <div className="metagraph-detail-row">
              <span>Recency</span>
              <strong>{f2(selectedPrec._rec)}</strong>
            </div>
            <div className="metagraph-detail-row">
              <span>Confidence</span>
              <strong>{f2(selectedPrec._conf)}</strong>
            </div>
            {selectedPrec.score != null && (
              <div className="metagraph-detail-row">
                <span>Search score</span>
                <strong>{f2(selectedPrec.score)}</strong>
              </div>
            )}
            {selectedPrec.url && (
              <div className="metagraph-detail-row">
                <span>URL</span>
                <a href={selectedPrec.url} target="_blank" rel="noopener noreferrer" style={{ fontSize: '0.75rem', color: '#7c3aed' }}>
                  Apri ↗
                </a>
              </div>
            )}
            {selectedPrec.summary && (
              <>
                <h6 className="metagraph-detail-sub">Sommario</h6>
                <p style={{ fontSize: '0.78rem', color: '#374151', lineHeight: 1.4, margin: '4px 0' }}>
                  {selectedPrec.summary.slice(0, 400)}{selectedPrec.summary.length > 400 ? '…' : ''}
                </p>
              </>
            )}
            <h6 className="metagraph-detail-sub">Formula Δ</h6>
            <p style={{ fontSize: '0.78rem', color: '#374151', fontFamily: 'monospace' }}>
              δ = bind({f2(selectedPrec._bind)}) × sim({f2(selectedPrec._sim)}) × rec({f2(selectedPrec._rec)}) × stance({selectedPrec._stance}) × conf({f2(selectedPrec._conf)})
              = <strong>{selectedPrec._delta >= 0 ? '+' : ''}{f2(selectedPrec._delta)}</strong>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
