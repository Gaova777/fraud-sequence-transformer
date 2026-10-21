import { useEffect, useMemo, useState } from 'react'

const API = 'http://localhost:8000'
const MODELS = [
  { key: 'transformer', label: 'Transformer (sequence)' },
  { key: 'lstm', label: 'LSTM (sequence)' },
  { key: 'mlp', label: 'MLP (sequence)' },
  { key: 'xgboost', label: 'XGBoost (per transaction)' },
]

interface Tx {
  user: number
  ts: string
  amount: number
  city: string
  state: string | null
  mcc: string
  chip: string
  risk: number
  fraud: number
  at_stake: number
}

interface Point {
  threshold: number
  alerts: number
  recall: number
  cost: number
}

interface Cost {
  fee: number
  no_model_cost: number
  best: Point
  curve: Point[]
}

const money = (x: number) => x.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })

function CostChart({ cost, threshold }: { cost: Cost; threshold: number }) {
  const w = 520
  const h = 170
  const pts = cost.curve
  const max = Math.max(...pts.map((p) => p.cost), cost.no_model_cost)
  const x = (i: number) => 30 + (i / (pts.length - 1)) * (w - 40)
  const y = (c: number) => 10 + (1 - c / max) * (h - 30)
  const bestI = pts.findIndex((p) => p.threshold === cost.best.threshold)
  const selI = pts.reduce(
    (b, p, i) => (Math.abs(p.threshold - threshold) < Math.abs(pts[b].threshold - threshold) ? i : b),
    0,
  )
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart" role="img" aria-label="Cost by threshold">
      <line x1={30} x2={w - 10} y1={y(cost.no_model_cost)} y2={y(cost.no_model_cost)} className="ref" />
      <text x={w - 12} y={y(cost.no_model_cost) - 4} className="lbl" textAnchor="end">
        no model: {money(cost.no_model_cost)}
      </text>
      <polyline points={pts.map((p, i) => `${x(i)},${y(p.cost)}`).join(' ')} className="curve" />
      <circle cx={x(bestI)} cy={y(cost.best.cost)} r={5} className="best" />
      <line x1={x(selI)} x2={x(selI)} y1={10} y2={h - 20} className="sel" />
      <text x={30} y={h - 4} className="lbl">
        stricter threshold (fewer alerts) →
      </text>
    </svg>
  )
}

export default function App() {
  const [model, setModel] = useState('transformer')
  const [fee, setFee] = useState(25)
  const [stream, setStream] = useState<Tx[]>([])
  const [cost, setCost] = useState<Cost | null>(null)
  const [threshold, setThreshold] = useState<number | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API}/stream?model=${model}&n=300`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then(setStream)
      .catch(() => setError(`No scores for ${model} yet, or the API is not running.`))
  }, [model])

  useEffect(() => {
    fetch(`${API}/cost?model=${model}&fee=${fee}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((c: Cost) => {
        setCost(c)
        setThreshold(c.best.threshold)
        setError('')
      })
      .catch(() => setCost(null))
  }, [model, fee])

  const sel = useMemo(() => {
    if (!cost || threshold === null) return null
    return cost.curve.reduce((b, p) => (Math.abs(p.threshold - threshold) < Math.abs(b.threshold - threshold) ? p : b))
  }, [cost, threshold])

  const queue = useMemo(() => [...stream].sort((a, b) => b.risk - a.risk), [stream])

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo" />
          Fraud operations
        </div>
        <span className="sub">Card transactions · scored from each customer's last ten purchases</span>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {MODELS.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
      </header>

      {error && <p className="error">{error}</p>}

      <section className="kpis">
        <div>
          <span>Cost of reviewing one alert</span>
          <b>{money(fee)}</b>
          <input type="range" min={1} max={200} value={fee} onChange={(e) => setFee(+e.target.value)} />
        </div>
        <div>
          <span>Alerts to review (test period)</span>
          <b>{sel ? sel.alerts.toLocaleString() : '—'}</b>
        </div>
        <div>
          <span>Fraud caught</span>
          <b className="good">{sel ? `${Math.round(sel.recall * 100)}%` : '—'}</b>
        </div>
        <div>
          <span>Total cost at this threshold</span>
          <b>{sel ? money(sel.cost) : '—'}</b>
        </div>
        <div>
          <span>Saved vs. no model</span>
          <b className="good">{sel && cost ? money(cost.no_model_cost - sel.cost) : '—'}</b>
        </div>
      </section>

      <main className="grid">
        <section className="panel">
          <header>
            <h2>Cost by alert threshold</h2>
            <span className="muted">green dot: cheapest threshold for this review cost</span>
          </header>
          {cost && threshold !== null ? (
            <>
              <CostChart cost={cost} threshold={threshold} />
              <input
                className="th"
                type="range"
                min={0}
                max={cost.curve.length - 1}
                value={cost.curve.findIndex((p) => p === sel)}
                onChange={(e) => setThreshold(cost.curve[+e.target.value].threshold)}
              />
              <p className="muted small">
                A missed fraud costs what was stolen; every alert costs the review fee, real fraud or not. Move the fee
                and the cheapest threshold moves with it: when reviews are cheap, it pays to look at more.
              </p>
            </>
          ) : (
            <p className="muted pad">Loading…</p>
          )}
        </section>

        <section className="panel queue">
          <header>
            <h2>Alert queue</h2>
            <span className="muted">replay of {stream.length} test windows, highest risk first</span>
          </header>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Risk</th>
                  <th>Customer</th>
                  <th>Last purchase</th>
                  <th>Where</th>
                  <th>Amount</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {queue.map((t, i) => {
                  const flagged = threshold !== null && t.risk >= threshold
                  return (
                    <tr key={i} className={flagged ? 'flag' : ''}>
                      <td>
                        <span className="bar">
                          <i style={{ width: `${Math.max(2, t.risk * 100)}%` }} />
                        </span>
                        {(t.risk * 100).toFixed(1)}%
                      </td>
                      <td className="mono">#{t.user}</td>
                      <td>{new Date(t.ts).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}</td>
                      <td>
                        {t.chip === 'Online Transaction' ? 'online' : `${t.city}${t.state ? ', ' + t.state : ''}`}
                      </td>
                      <td className="mono">{money(t.amount)}</td>
                      <td>
                        {t.fraud ? (
                          <span className={'tag ' + (flagged ? 'caught' : 'missed')}>
                            fraud {flagged ? 'caught' : 'missed'} · {money(t.at_stake)}
                          </span>
                        ) : flagged ? (
                          <span className="tag fa">false alarm</span>
                        ) : (
                          <span className="tag ok">legit</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  )
}
