import { useState, useMemo } from 'react';

/**
 * AttackTextDetails – shows completed cross-attacks with full link text.
 *
 * For every active attack (not filtered, value > 0) it renders:
 *   ATTACKER link  ──⚔️──▶  TARGET link
 * with the full premise/conclusion text of both sides, plus metadata.
 */
export default function AttackTextDetails({ aqaReport }) {
  const [expandedIdx, setExpandedIdx] = useState(null);

  const attacks = useMemo(() => {
    if (!aqaReport?.links) return [];

    const proLinks = aqaReport.links.pro ?? [];
    const contraLinks = aqaReport.links.contra ?? [];
    const allLinks = [...proLinks, ...contraLinks];

    // Build link_id → link map (with role info)
    const linkMap = {};
    for (const link of allLinks) {
      if (link.link_id) linkMap[link.link_id] = link;
    }

    // Collect every active attack
    const result = [];
    for (const targetLink of allLinks) {
      const received = targetLink.attacks_received ?? [];
      for (const atk of received) {
        if (atk.filtered) continue;
        if ((atk.attack_value ?? 0) <= 0) continue;

        const attackerLink = linkMap[atk.attacker_link_id] ?? null;
        result.push({
          target: targetLink,
          attacker: attackerLink,
          attackerLinkId: atk.attacker_link_id,
          attackerRole: atk.attacker_role,
          attackValue: atk.attack_value ?? 0,
          overlap: atk.overlap ?? 0,
          attackType: atk.attack_type ?? 'unknown',
          nliLabel: atk.nli_label ?? '',
          nliBypass: atk.nli_bypass ?? false,
          typeMultiplier: atk.type_multiplier ?? 1.0,
        });
      }
    }

    // Sort by attack value descending
    result.sort((a, b) => b.attackValue - a.attackValue);
    return result;
  }, [aqaReport]);

  if (!attacks.length) return null;

  const roleLabel = (role) => {
    if (!role) return '?';
    const r = role.toLowerCase();
    if (r === 'support' || r === 'pro') return 'PRO';
    if (r === 'counter' || r === 'contra') return 'CONTRO';
    return role.toUpperCase();
  };

  const roleClass = (role) => {
    const r = (role || '').toLowerCase();
    if (r === 'support' || r === 'pro') return 'atk-role-pro';
    return 'atk-role-contra';
  };

  const toggle = (idx) => setExpandedIdx(expandedIdx === idx ? null : idx);

  return (
    <div className="attack-text-details">
      <p className="attack-text-summary">
        {attacks.length} attacco{attacks.length !== 1 ? 'hi' : ''} attivo{attacks.length !== 1 ? 'i' : ''} completato{attacks.length !== 1 ? 'i' : ''}
      </p>

      {attacks.map((atk, idx) => {
        const expanded = expandedIdx === idx;
        const targetRole = roleLabel(atk.target.role);
        const attackerRole = roleLabel(atk.attackerRole);

        return (
          <div key={idx} className="attack-text-card" onClick={() => toggle(idx)}>
            {/* Compact header */}
            <div className="attack-text-header">
              <div className="attack-text-sides">
                <span className={`atk-role-tag ${roleClass(atk.attackerRole)}`}>
                  {attackerRole}
                </span>
                <strong>{atk.attackerLinkId}</strong>
                <span className="attack-arrow">⚔️ →</span>
                <span className={`atk-role-tag ${roleClass(atk.target.role)}`}>
                  {targetRole}
                </span>
                <strong>{atk.target.link_id}</strong>
              </div>
              <div className="attack-text-meta-inline">
                <span className="atk-val">danno {atk.attackValue.toFixed(4)}</span>
                <span className="atk-overlap">overlap {(atk.overlap * 100).toFixed(0)}%</span>
                <span className="atk-type">{atk.attackType.replace('_', ' ')}</span>
                {atk.nliBypass && <span className="atk-nli-badge">NLI bypass</span>}
              </div>
              <span className={`attack-expand-icon ${expanded ? 'open' : ''}`}>▸</span>
            </div>

            {/* Expanded: full text */}
            {expanded && (
              <div className="attack-text-body">
                {/* Attacker */}
                <div className="attack-text-block attacker-block">
                  <div className="attack-text-block-label">
                    <span className={`atk-role-tag ${roleClass(atk.attackerRole)}`}>
                      {attackerRole}
                    </span>
                    Attaccante — {atk.attackerLinkId}
                  </div>
                  {atk.attacker ? (
                    <>
                      <div className="attack-text-segment">
                        <span className="segment-label">Premessa:</span>
                        <p>{atk.attacker.premise_text || '—'}</p>
                      </div>
                      <div className="attack-text-segment">
                        <span className="segment-label">Conclusione:</span>
                        <p>{atk.attacker.conclusion_text || '—'}</p>
                      </div>
                    </>
                  ) : (
                    <p className="attack-text-missing">Testo del link non disponibile</p>
                  )}
                </div>

                {/* Target */}
                <div className="attack-text-block target-block">
                  <div className="attack-text-block-label">
                    <span className={`atk-role-tag ${roleClass(atk.target.role)}`}>
                      {targetRole}
                    </span>
                    Bersaglio — {atk.target.link_id}
                  </div>
                  <div className="attack-text-segment">
                    <span className="segment-label">Premessa:</span>
                    <p>{atk.target.premise_text || '—'}</p>
                  </div>
                  <div className="attack-text-segment">
                    <span className="segment-label">Conclusione:</span>
                    <p>{atk.target.conclusion_text || '—'}</p>
                  </div>
                </div>

                {/* Metadata row */}
                <div className="attack-text-meta-row">
                  <span>Tipo: <strong>{atk.attackType}</strong></span>
                  <span>Moltiplicatore: <strong>×{atk.typeMultiplier.toFixed(1)}</strong></span>
                  <span>NLI: <strong>{atk.nliLabel || '—'}</strong>{atk.nliBypass ? ' (bypass)' : ''}</span>
                  <span>Overlap: <strong>{(atk.overlap * 100).toFixed(1)}%</strong></span>
                  <span>Danno: <strong>{atk.attackValue.toFixed(4)}</strong></span>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
