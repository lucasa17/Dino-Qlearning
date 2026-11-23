export const ACTION_RUN = 0;
export const ACTION_JUMP = 1;

export const ACTION_LABELS = {
  [ACTION_RUN]: 'Correr',
  [ACTION_JUMP]: 'Saltar',
};

export const DIST_BINS = [
  { id: 'D0', short: 'D0', name: 'Iminente', range: '≤ 45 px', max: 45 },
  { id: 'D1', short: 'D1', name: 'Crítica', range: '46–95 px', max: 95 },
  { id: 'D2', short: 'D2', name: 'Curta', range: '96–145 px', max: 145 },
  { id: 'D3', short: 'D3', name: 'Longa', range: '146–210 px', max: 210 },
  { id: 'D4', short: 'D4', name: 'Distante', range: '211–300 px', max: 300 },
  { id: 'D5', short: 'D5', name: 'Livre', range: '> 300 px', max: Infinity },
];

export const SPEED_BINS = [
  { id: 'V0', name: 'Normal', range: '< 7.3 px/f', threshold: 7.3 },
  { id: 'V1', name: 'Alta', range: '≥ 7.3 px/f', threshold: Infinity },
];

export const STATES = [];
for (let s = 0; s < SPEED_BINS.length; s++) {
  for (let d = 0; d < DIST_BINS.length; d++) {
    const dist = DIST_BINS[d];
    const spd = SPEED_BINS[s];
    STATES.push({
      index: s * DIST_BINS.length + d,
      distBin: d,
      speedBin: s,
      code: `${dist.id} · ${spd.name}`,
      distLabel: `${dist.id} · ${dist.name}`,
      rangeLabel: `${dist.range} · ${spd.range}`,
      distRange: dist.range,
      speedName: spd.name,
    });
  }
}

export function discretizeState(distancePx, speedPxPerFrame) {
  let distBin = DIST_BINS.length - 1;
  for (let i = 0; i < DIST_BINS.length; i++) {
    if (distancePx <= DIST_BINS[i].max) {
      distBin = i;
      break;
    }
  }
  const speedBin = speedPxPerFrame < SPEED_BINS[0].threshold ? 0 : 1;
  return speedBin * DIST_BINS.length + distBin;
}

export class QLearningAgent {
  constructor() {
    this.alpha = 0.25;
    this.gamma = 0.9;
    this.epsilon = 0.6;
    this.epsilonMin = 0.02;
    this.epsilonDecay = 0.975;
    this.autoDecay = true;
    this.resetMemory();
  }

  resetMemory() {
    this.qTable = STATES.map(() => [0, 0]);
    this.visits = STATES.map(() => [0, 0]);
    this.lastUpdate = null;
    this.stateLastUpdate = new Array(STATES.length).fill(null);
    this.epsilon = 0.6;
  }

  selectAction(stateIdx, forceGreedy = false) {
    const qValues = this.qTable[stateIdx];
    const qRun = qValues[ACTION_RUN];
    const qJump = qValues[ACTION_JUMP];

    if (!forceGreedy && Math.random() < this.epsilon) {
      const action = Math.random() < 0.5 ? ACTION_RUN : ACTION_JUMP;
      return {
        action,
        isExploration: true,
        sourceLabel: 'Exploração (ε)',
      };
    }

    let action;
    if (Math.abs(qJump - qRun) < 1e-6) {
      action = Math.random() < 0.5 ? ACTION_RUN : ACTION_JUMP;
    } else {
      action = qJump > qRun ? ACTION_JUMP : ACTION_RUN;
    }

    return {
      action,
      isExploration: false,
      sourceLabel: forceGreedy ? 'Política Pura' : 'Política Q',
    };
  }

  getBestAction(stateIdx) {
    const [qRun, qJump] = this.qTable[stateIdx];
    if (Math.abs(qJump - qRun) < 1e-6) {
      return null;
    }
    return qJump > qRun ? ACTION_JUMP : ACTION_RUN;
  }

  update(stateIdx, action, reward, nextStateIdx, isTerminal) {
    const oldQ = this.qTable[stateIdx][action];
    const nextQ = this.qTable[nextStateIdx];
    const maxNextQ = isTerminal ? 0 : Math.max(nextQ[ACTION_RUN], nextQ[ACTION_JUMP]);
    const tdTarget = reward + this.gamma * maxNextQ;
    const tdError = tdTarget - oldQ;
    const newQ = oldQ + this.alpha * tdError;

    this.qTable[stateIdx][action] = newQ;
    this.visits[stateIdx][action] += 1;

    const record = {
      stateIdx,
      action,
      reward,
      nextStateIdx,
      isTerminal,
      oldQ,
      maxNextQ,
      tdTarget,
      tdError,
      newQ,
      alpha: this.alpha,
      gamma: this.gamma,
      delta: newQ - oldQ,
      timestamp: performance.now(),
    };

    this.lastUpdate = record;
    this.stateLastUpdate[stateIdx] = record;
    return record;
  }

  decayEpsilon() {
    if (!this.autoDecay) return;
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }
}
