import {
  ACTION_JUMP,
  ACTION_LABELS,
  ACTION_RUN,
  DIST_BINS,
  QLearningAgent,
  SPEED_BINS,
  STATES,
} from './qlearning.js';
import { DinoEnvironment } from './game.js';
import { LearningChart } from './charts.js';

const agent = new QLearningAgent();
const gameCanvas = document.getElementById('game-canvas');
const chartCanvas = document.getElementById('chart-canvas');
const chart = new LearningChart(chartCanvas);

// State UI elements
const elEpisode = document.getElementById('meta-episode');
const elScore = document.getElementById('meta-score');
const elBest = document.getElementById('meta-best');
const elSpeed = document.getElementById('meta-speed');
const elEpsilon = document.getElementById('meta-epsilon');

const elReadoutState = document.getElementById('readout-state');
const elReadoutDistance = document.getElementById('readout-distance');
const elReadoutAction = document.getElementById('readout-action');
const elReadoutSource = document.getElementById('readout-source');

const elBellmanDelta = document.getElementById('bellman-delta-label');
const elFormulaLive = document.getElementById('formula-live');
const elBsa = document.getElementById('b-sa');
const elBreward = document.getElementById('b-reward');
const elBnext = document.getElementById('b-next');
const elBqchange = document.getElementById('b-qchange');

const elStatAvg10 = document.getElementById('stat-avg10');
const elStatEpCount = document.getElementById('stat-epcount');

const elPolicyMatrix = document.getElementById('policy-matrix');
const elQTableBody = document.getElementById('qtable-body');

// Controls
const btnPlayPause = document.getElementById('btn-play-pause');
const btnStep = document.getElementById('btn-step');
const btnBatchTrain = document.getElementById('btn-batch-train');
const btnHeaderReset = document.getElementById('btn-header-reset');

const sliderAlpha = document.getElementById('slider-alpha');
const sliderGamma = document.getElementById('slider-gamma');
const sliderEpsilon = document.getElementById('slider-epsilon');
const valAlpha = document.getElementById('val-alpha');
const valGamma = document.getElementById('val-gamma');
const valEpsilon = document.getElementById('val-epsilon');
const chkAutoDecay = document.getElementById('chk-auto-decay');

let isRunning = true;
let simSpeed = 1;
let stepPending = false;
let inspectedStateIdx = null;
let lastFlashedCell = { stateIdx: -1, action: -1, token: 0 };

// Build Policy Matrix DOM once
const policyCells = [];
function buildPolicyMatrix() {
  elPolicyMatrix.innerHTML = '';

  const corner = document.createElement('div');
  corner.className = 'pm-head';
  corner.textContent = 'Vel \\ Dist';
  elPolicyMatrix.appendChild(corner);

  for (const dist of DIST_BINS) {
    const h = document.createElement('div');
    h.className = 'pm-head';
    h.textContent = dist.short;
    h.title = `${dist.name} (${dist.range})`;
    elPolicyMatrix.appendChild(h);
  }

  for (let s = 0; s < SPEED_BINS.length; s++) {
    const spd = SPEED_BINS[s];
    const rowLabel = document.createElement('div');
    rowLabel.className = 'pm-row-label';
    rowLabel.innerHTML = `<span>${spd.name}</span><small>${spd.range}</small>`;
    elPolicyMatrix.appendChild(rowLabel);

    for (let d = 0; d < DIST_BINS.length; d++) {
      const idx = s * DIST_BINS.length + d;
      const cell = document.createElement('div');
      cell.className = 'pm-cell';
      cell.setAttribute('role', 'gridcell');
      cell.innerHTML = `<span class="pm-icon">—</span><span class="pm-label">Neutro</span>`;
      elPolicyMatrix.appendChild(cell);
      policyCells[idx] = cell;
    }
  }
}

// Build Q-Table 12 rows DOM once for smooth, non-flickering real-time updates
const qRowRefs = [];
function buildQTable() {
  elQTableBody.innerHTML = '';
  for (const st of STATES) {
    const tr = document.createElement('tr');
    if (st.distBin === 0 && st.speedBin > 0) {
      tr.classList.add('row-group-start');
    }
    tr.dataset.stateIdx = String(st.index);

    tr.innerHTML = `
      <td>
        <div class="state-cell">
          <span class="state-name">${st.code}</span>
          <span class="state-range">${st.distRange}</span>
        </div>
      </td>
      <td class="num" data-col="run">
        <div class="q-cell-wrap">
          <span class="q-val zero">0.00</span>
          <div class="q-bar-track"><div class="q-bar-fill pos"></div></div>
        </div>
      </td>
      <td class="num" data-col="jump">
        <div class="q-cell-wrap">
          <span class="q-val zero">0.00</span>
          <div class="q-bar-track"><div class="q-bar-fill jump-pos"></div></div>
        </div>
      </td>
      <td data-col="policy">
        <span class="policy-text tie">— Indefinido</span>
      </td>
      <td class="num visits-val" data-col="visits">0</td>
    `;

    tr.addEventListener('click', () => {
      inspectedStateIdx = inspectedStateIdx === st.index ? null : st.index;
      renderBellmanInspector();
      updateQTableUI();
    });

    elQTableBody.appendChild(tr);

    qRowRefs[st.index] = {
      tr,
      runCell: tr.querySelector('[data-col="run"]'),
      runVal: tr.querySelector('[data-col="run"] .q-val'),
      runBar: tr.querySelector('[data-col="run"] .q-bar-fill'),
      jumpCell: tr.querySelector('[data-col="jump"]'),
      jumpVal: tr.querySelector('[data-col="jump"] .q-val'),
      jumpBar: tr.querySelector('[data-col="jump"] .q-bar-fill'),
      policyEl: tr.querySelector('[data-col="policy"] .policy-text'),
      visitsEl: tr.querySelector('[data-col="visits"]'),
    };
  }
}

function formatQ(v) {
  const sign = v > 0.005 ? '+' : '';
  return `${sign}${v.toFixed(2)}`;
}

function triggerCellFlash(stateIdx, action, delta) {
  const ref = qRowRefs[stateIdx];
  if (!ref) return;
  const cell = action === ACTION_RUN ? ref.runCell : ref.jumpCell;
  cell.classList.remove('flash-pos', 'flash-neg');
  // Force reflow so rapid consecutive updates still flash visibly
  void cell.offsetWidth;
  cell.classList.add(delta >= 0 ? 'flash-pos' : 'flash-neg');
}

function updateQTableUI() {
  const currentStateIdx = env.getCurrentStateIndex();

  // Compute max magnitude for proportional bar scaling
  let maxMag = 25;
  for (let i = 0; i < STATES.length; i++) {
    maxMag = Math.max(maxMag, Math.abs(agent.qTable[i][0]), Math.abs(agent.qTable[i][1]));
  }

  for (let i = 0; i < STATES.length; i++) {
    const ref = qRowRefs[i];
    const [qRun, qJump] = agent.qTable[i];
    const [vRun, vJump] = agent.visits[i];

    ref.tr.classList.toggle('is-current', i === currentStateIdx);
    ref.tr.classList.toggle('is-inspected', i === inspectedStateIdx);

    // Run value & bar
    ref.runVal.textContent = formatQ(qRun);
    ref.runVal.className = `q-val ${qRun > 0.01 ? 'pos' : qRun < -0.01 ? 'neg' : 'zero'}`;
    ref.runBar.style.width = `${Math.min(100, (Math.abs(qRun) / maxMag) * 100)}%`;
    ref.runBar.className = `q-bar-fill ${qRun >= 0 ? 'pos' : 'neg'}`;

    // Jump value & bar
    ref.jumpVal.textContent = formatQ(qJump);
    ref.jumpVal.className = `q-val ${qJump > 0.01 ? 'pos' : qJump < -0.01 ? 'neg' : 'zero'}`;
    ref.jumpBar.style.width = `${Math.min(100, (Math.abs(qJump) / maxMag) * 100)}%`;
    ref.jumpBar.className = `q-bar-fill ${qJump >= 0 ? 'jump-pos' : 'neg'}`;

    // Best policy
    const best = agent.getBestAction(i);
    if (best === ACTION_RUN) {
      ref.policyEl.textContent = '→ Correr';
      ref.policyEl.className = 'policy-text run';
    } else if (best === ACTION_JUMP) {
      ref.policyEl.textContent = '▲ Saltar';
      ref.policyEl.className = 'policy-text jump';
    } else {
      ref.policyEl.textContent = '— Neutro';
      ref.policyEl.className = 'policy-text tie';
    }

    ref.visitsEl.textContent = String(vRun + vJump);

    // Policy matrix cell
    const pmCell = policyCells[i];
    pmCell.classList.toggle('is-active', i === currentStateIdx);
    pmCell.classList.remove('act-run', 'act-jump');
    if (best === ACTION_RUN) {
      pmCell.classList.add('act-run');
      pmCell.innerHTML = `<span class="pm-icon">→</span><span class="pm-label">Correr</span>`;
    } else if (best === ACTION_JUMP) {
      pmCell.classList.add('act-jump');
      pmCell.innerHTML = `<span class="pm-icon">▲</span><span class="pm-label">Saltar</span>`;
    } else {
      pmCell.innerHTML = `<span class="pm-icon">—</span><span class="pm-label">Neutro</span>`;
    }
  }
}

function renderBellmanInspector() {
  const rec =
    inspectedStateIdx !== null
      ? agent.stateLastUpdate[inspectedStateIdx]
      : agent.lastUpdate;

  if (!rec) {
    elBellmanDelta.textContent = 'ΔQ 0.00';
    elBellmanDelta.className = 'delta-readout mono';
    elFormulaLive.textContent =
      inspectedStateIdx !== null
        ? `Estado ${STATES[inspectedStateIdx].code} ainda não foi atualizado neste treino.`
        : 'Aguardando primeira transição de estado...';
    elBsa.textContent = '—';
    elBreward.textContent = '—';
    elBnext.textContent = '—';
    elBqchange.textContent = '—';
    return;
  }

  const sLabel = STATES[rec.stateIdx].code;
  const aLabel = ACTION_LABELS[rec.action];
  const nextLabel = rec.isTerminal ? 'Terminal' : STATES[rec.nextStateIdx].code;
  const deltaSign = rec.delta >= 0 ? '+' : '';

  elBellmanDelta.textContent = `ΔQ ${deltaSign}${rec.delta.toFixed(2)}`;
  elBellmanDelta.className = `delta-readout mono ${rec.delta >= 0 ? 'pos' : 'neg'}`;

  elFormulaLive.textContent =
    `Q(${sLabel}, ${aLabel}) ← ${rec.oldQ.toFixed(2)} + ${rec.alpha.toFixed(2)} · ` +
    `[ ${rec.reward.toFixed(1)} + ${rec.gamma.toFixed(2)} · ${rec.maxNextQ.toFixed(2)} − ${rec.oldQ.toFixed(2)} ] ` +
    `= ${rec.newQ.toFixed(2)}`;

  elBsa.textContent = `${sLabel} → ${aLabel}`;
  elBreward.textContent = `${rec.reward >= 0 ? '+' : ''}${rec.reward.toFixed(1)}`;
  elBnext.textContent = rec.isTerminal ? 'Terminal (0.00)' : `${nextLabel} (máx ${rec.maxNextQ.toFixed(2)})`;
  elBqchange.textContent = `${rec.oldQ.toFixed(2)} → ${rec.newQ.toFixed(2)}`;
}

function updateLiveReadouts() {
  const stateIdx = env.getCurrentStateIndex();
  const st = STATES[stateIdx];
  const dist = env.getDistanceToNextObstacle();

  elEpisode.textContent = String(env.episode);
  elScore.textContent = String(Math.floor(env.score));
  elBest.textContent = String(env.bestScore);
  elSpeed.textContent = `${env.speed.toFixed(1)} px/f`;
  elEpsilon.textContent = agent.epsilon.toFixed(2);

  sliderEpsilon.value = agent.epsilon.toFixed(2);
  valEpsilon.textContent = agent.epsilon.toFixed(2);

  elReadoutState.textContent = st.code;
  elReadoutDistance.textContent = dist > 500 ? '> 500 px' : `${dist} px`;

  const dec = env.lastDecision;
  if (dec) {
    elReadoutAction.textContent = dec.action === ACTION_JUMP ? '▲ Saltar' : '→ Correr';
    if (env.mode === 'human') {
      const qRec = agent.getBestAction(stateIdx);
      const sug = qRec === null ? 'Neutro' : ACTION_LABELS[qRec];
      elReadoutSource.textContent = `Humano (Q sugere: ${sug})`;
    } else {
      elReadoutSource.textContent = dec.sourceLabel;
    }
  }
}

function updateChartAndStats() {
  chart.render(env.history);
  const n = env.history.length;
  elStatEpCount.textContent = String(n);
  if (n === 0) {
    elStatAvg10.textContent = '0';
    return;
  }
  const slice = env.history.slice(-10);
  const avg = slice.reduce((acc, item) => acc + item.score, 0) / slice.length;
  elStatAvg10.textContent = avg.toFixed(0);
}

const env = new DinoEnvironment(gameCanvas, agent, {
  onTransition(record) {
    lastFlashedCell = {
      stateIdx: record.stateIdx,
      action: record.action,
      token: lastFlashedCell.token + 1,
    };
    if (simSpeed <= 10 || stepPending) {
      triggerCellFlash(record.stateIdx, record.action, record.delta);
    }
    renderBellmanInspector();

    if (stepPending) {
      stepPending = false;
      setRunning(false);
    }
  },
  onEpisodeEnd() {
    updateChartAndStats();
  },
});

function setRunning(running) {
  isRunning = running;
  btnPlayPause.textContent = isRunning ? 'Pausar' : 'Iniciar';
}

// Mode segmented control
document.getElementById('mode-segmented').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-mode]');
  if (!btn) return;
  const mode = btn.dataset.mode;
  env.mode = mode;
  for (const b of document.querySelectorAll('#mode-segmented .seg-btn')) {
    b.classList.toggle('active', b === btn);
  }
  env.resetEpisode();
  updateLiveReadouts();
  updateQTableUI();
});

// Speed segmented control
document.getElementById('speed-segmented').addEventListener('click', (e) => {
  const btn = e.target.closest('[data-speed]');
  if (!btn) return;
  simSpeed = Number(btn.dataset.speed);
  for (const b of document.querySelectorAll('#speed-segmented .seg-btn')) {
    b.classList.toggle('active', b === btn);
  }
});

// Playback buttons
btnPlayPause.addEventListener('click', () => {
  stepPending = false;
  setRunning(!isRunning);
});

btnStep.addEventListener('click', () => {
  if (env.mode !== 'train') {
    // Allow stepping even in eval/human by advancing one frame
    env.stepFrame(true);
    env.render();
    updateLiveReadouts();
    updateQTableUI();
    return;
  }
  stepPending = true;
  simSpeed = 1;
  for (const b of document.querySelectorAll('#speed-segmented .seg-btn')) {
    b.classList.toggle('active', b.dataset.speed === '1');
  }
  setRunning(true);
});

btnBatchTrain.addEventListener('click', () => {
  env.runHeadlessEpisodes(25);
  env.resetEpisode();
  env.render();
  updateLiveReadouts();
  updateQTableUI();
  renderBellmanInspector();
  updateChartAndStats();
});

btnHeaderReset.addEventListener('click', () => {
  agent.resetMemory();
  env.episode = 1;
  env.bestScore = 0;
  env.history = [];
  inspectedStateIdx = null;
  env.resetEpisode();
  env.render();
  updateLiveReadouts();
  updateQTableUI();
  renderBellmanInspector();
  updateChartAndStats();
});

// Sliders
sliderAlpha.addEventListener('input', () => {
  agent.alpha = Number(sliderAlpha.value);
  valAlpha.textContent = agent.alpha.toFixed(2);
});

sliderGamma.addEventListener('input', () => {
  agent.gamma = Number(sliderGamma.value);
  valGamma.textContent = agent.gamma.toFixed(2);
});

sliderEpsilon.addEventListener('input', () => {
  agent.epsilon = Number(sliderEpsilon.value);
  valEpsilon.textContent = agent.epsilon.toFixed(2);
  elEpsilon.textContent = agent.epsilon.toFixed(2);
});

chkAutoDecay.addEventListener('change', () => {
  agent.autoDecay = chkAutoDecay.checked;
});

// Keyboard & Canvas click for Human mode
window.addEventListener('keydown', (e) => {
  if (env.mode === 'human' && (e.code === 'Space' || e.code === 'ArrowUp')) {
    e.preventDefault();
    env.requestHumanJump();
  }
});

gameCanvas.addEventListener('pointerdown', () => {
  if (env.mode === 'human') {
    env.requestHumanJump();
  }
});

// Main animation loop
let uiTickCounter = 0;
function loop() {
  if (isRunning) {
    const steps = stepPending ? 1 : simSpeed;
    for (let i = 0; i < steps; i++) {
      env.stepFrame(i === steps - 1 || simSpeed <= 3);
      if (!isRunning) break;
    }
    env.render();
    uiTickCounter += 1;

    // Keep Q-table & readouts smooth at 60fps in 1x-10x, or every 2 frames in 50x
    if (simSpeed < 50 || uiTickCounter % 2 === 0 || stepPending) {
      updateLiveReadouts();
      updateQTableUI();
    }
  }
  requestAnimationFrame(loop);
}

// Initialize DOM & start loop
buildPolicyMatrix();
buildQTable();
env.render();
updateLiveReadouts();
updateQTableUI();
renderBellmanInspector();
updateChartAndStats();
requestAnimationFrame(loop);
