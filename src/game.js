import {
  ACTION_JUMP,
  ACTION_LABELS,
  ACTION_RUN,
  DIST_BINS,
  STATES,
  discretizeState,
} from './qlearning.js';

const LOGICAL_WIDTH = 840;
const LOGICAL_HEIGHT = 260;
const GROUND_Y = 212;
const DINO_X = 84;
const DINO_W = 34;
const DINO_H = 44;
const JUMP_VELOCITY = -12.8;
const GRAVITY = 0.68;

const CACTUS_VARIANTS = [
  { w: 22, h: 36, type: 'small' },
  { w: 26, h: 48, type: 'tall' },
  { w: 42, h: 40, type: 'double' },
];

export class DinoEnvironment {
  constructor(canvas, agent, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.agent = agent;
    this.callbacks = callbacks;

    this.episode = 1;
    this.bestScore = 0;
    this.history = [];
    this.mode = 'train'; // 'train' | 'eval' | 'human'

    this.humanJumpBuffer = 0;
    this.floatingRewards = [];
    this.groundPebbles = Array.from({ length: 28 }, (_, i) => ({
      x: i * 34 + (i % 3) * 7,
      w: (i % 4) + 2,
      yOffset: (i % 3) * 5 + 8,
    }));
    this.clouds = [
      { x: 180, y: 48, w: 54 },
      { x: 460, y: 34, w: 68 },
      { x: 710, y: 58, w: 48 },
    ];

    this.resetEpisode();
    if (this.canvas) {
      this.configureCanvasDpi();
    }
  }

  configureCanvasDpi() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = LOGICAL_WIDTH * dpr;
    this.canvas.height = LOGICAL_HEIGHT * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  resetEpisode() {
    this.dino = {
      x: DINO_X,
      y: GROUND_Y - DINO_H,
      w: DINO_W,
      h: DINO_H,
      vy: 0,
      isGrounded: true,
      legFrame: 0,
    };
    this.speed = 6.0;
    this.score = 0;
    this.frameCount = 0;
    this.obstacles = [];
    this.spawnInitialObstacle();

    this.pendingRun = null;
    this.pendingJump = null;
    this.lastDecision = {
      action: ACTION_RUN,
      isExploration: false,
      sourceLabel: 'Início',
      stateIdx: this.getCurrentStateIndex(),
    };
    this.floatingRewards = [];
    this.isDead = false;
  }

  spawnInitialObstacle() {
    const variant = CACTUS_VARIANTS[0];
    this.obstacles.push({
      x: 470 + Math.random() * 90,
      y: GROUND_Y - variant.h,
      w: variant.w,
      h: variant.h,
      type: variant.type,
      passed: false,
    });
  }

  maybeSpawnObstacle() {
    const last = this.obstacles[this.obstacles.length - 1];
    const minGap = 310 + this.speed * 15;
    const maxGap = minGap + 180;
    if (!last || LOGICAL_WIDTH - (last.x + last.w) >= minGap + Math.random() * (maxGap - minGap)) {
      if (!last || last.x < LOGICAL_WIDTH - 240) {
        const variant = CACTUS_VARIANTS[Math.floor(Math.random() * CACTUS_VARIANTS.length)];
        const spawnX = Math.max(LOGICAL_WIDTH + 30, last ? last.x + last.w + minGap + Math.random() * 140 : LOGICAL_WIDTH + 40);
        if (spawnX < LOGICAL_WIDTH + 220) {
          this.obstacles.push({
            x: spawnX,
            y: GROUND_Y - variant.h,
            w: variant.w,
            h: variant.h,
            type: variant.type,
            passed: false,
          });
        }
      }
    }
  }

  getNextObstacle() {
    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i];
      if (obs.x + obs.w >= this.dino.x + 6) {
        return obs;
      }
    }
    return null;
  }

  getDistanceToNextObstacle() {
    const nextObs = this.getNextObstacle();
    if (!nextObs) return 600;
    return Math.round(nextObs.x - (this.dino.x + this.dino.w));
  }

  getCurrentStateIndex() {
    return discretizeState(this.getDistanceToNextObstacle(), this.speed);
  }

  requestHumanJump() {
    this.humanJumpBuffer = 8;
  }

  addFloatingReward(value, x, y) {
    if (!this.ctx) return;
    const text = value >= 0 ? `+${value.toFixed(value % 1 === 0 ? 0 : 1)}` : `${value.toFixed(0)}`;
    this.floatingRewards.push({
      text,
      value,
      x,
      y,
      life: 44,
      maxLife: 44,
    });
  }

  checkCollision(obs) {
    // Small 4px inset hitbox for clean visual fairness
    const dx1 = this.dino.x + 5;
    const dx2 = this.dino.x + this.dino.w - 5;
    const dy1 = this.dino.y + 4;
    const dy2 = this.dino.y + this.dino.h - 3;

    const ox1 = obs.x + 3;
    const ox2 = obs.x + obs.w - 3;
    const oy1 = obs.y + 3;
    const oy2 = obs.y + obs.h;

    return dx1 < ox2 && dx2 > ox1 && dy1 < oy2 && dy2 > oy1;
  }

  stepFrame(recordVisuals = true) {
    let transitionOccurred = false;
    let episodeEnded = false;

    if (this.isDead) {
      this.episode += 1;
      if (this.mode === 'train') {
        this.agent.decayEpsilon();
      }
      this.resetEpisode();
      if (this.callbacks.onEpisodeStart) {
        this.callbacks.onEpisodeStart(this.episode);
      }
      return { transitionOccurred: false, episodeEnded: false };
    }

    this.frameCount += 1;
    this.speed = Math.min(9.2, 6.0 + this.score * 0.008);

    // 1. Complete pending RUN step if ready and still on ground
    if (this.dino.isGrounded && this.pendingRun) {
      const currentStateIdx = this.getCurrentStateIndex();
      if (this.pendingRun.frames >= 4 || currentStateIdx !== this.pendingRun.stateIdx) {
        const reward = 1.0 + this.pendingRun.clearedObstacles * 15.0;
        if (this.mode === 'train') {
          const updateRec = this.agent.update(
            this.pendingRun.stateIdx,
            ACTION_RUN,
            reward,
            currentStateIdx,
            false
          );
          if (this.callbacks.onTransition) {
            this.callbacks.onTransition(updateRec);
          }
          transitionOccurred = true;
        }
        this.pendingRun = null;
      }
    }

    // 2. Make a new decision if grounded and no pending action
    if (this.dino.isGrounded && !this.pendingRun && !this.pendingJump) {
      const stateIdx = this.getCurrentStateIndex();
      let decision;
      if (this.mode === 'human') {
        const shouldJump = this.humanJumpBuffer > 0;
        this.humanJumpBuffer = 0;
        decision = {
          action: shouldJump ? ACTION_JUMP : ACTION_RUN,
          isExploration: false,
          sourceLabel: 'Humano',
        };
      } else {
        decision = this.agent.selectAction(stateIdx, this.mode === 'eval');
      }

      this.lastDecision = {
        ...decision,
        stateIdx,
      };

      if (decision.action === ACTION_JUMP) {
        this.dino.vy = JUMP_VELOCITY;
        this.dino.isGrounded = false;
        this.pendingJump = {
          stateIdx,
          clearedObstacles: 0,
        };
      } else {
        this.pendingRun = {
          stateIdx,
          frames: 0,
          clearedObstacles: 0,
        };
      }
    }

    if (this.humanJumpBuffer > 0) {
      this.humanJumpBuffer -= 1;
    }

    // 3. Advance Dino vertical physics
    if (!this.dino.isGrounded) {
      this.dino.y += this.dino.vy;
      this.dino.vy += GRAVITY;
      if (this.dino.y >= GROUND_Y - this.dino.h) {
        this.dino.y = GROUND_Y - this.dino.h;
        this.dino.vy = 0;
        this.dino.isGrounded = true;
      }
    } else if (this.frameCount % 6 === 0) {
      this.dino.legFrame = 1 - this.dino.legFrame;
    }

    // 4. Move obstacles & world decor
    for (let i = 0; i < this.obstacles.length; i++) {
      const obs = this.obstacles[i];
      obs.x -= this.speed;

      if (!obs.passed && obs.x + obs.w < this.dino.x) {
        obs.passed = true;
        this.score += 10;
        if (this.pendingJump) {
          this.pendingJump.clearedObstacles += 1;
        } else if (this.pendingRun) {
          this.pendingRun.clearedObstacles += 1;
        }
      }
    }
    this.obstacles = this.obstacles.filter((obs) => obs.x + obs.w > -40);
    this.maybeSpawnObstacle();

    if (recordVisuals) {
      for (const p of this.groundPebbles) {
        p.x -= this.speed;
        if (p.x + p.w < 0) p.x = LOGICAL_WIDTH + Math.random() * 25;
      }
      for (const c of this.clouds) {
        c.x -= this.speed * 0.18;
        if (c.x + c.w < -10) c.x = LOGICAL_WIDTH + 30;
      }
      for (const f of this.floatingRewards) {
        f.y -= 0.7;
        f.life -= 1;
      }
      this.floatingRewards = this.floatingRewards.filter((f) => f.life > 0);
    }

    this.score += 0.12;
    if (Math.floor(this.score) > this.bestScore) {
      this.bestScore = Math.floor(this.score);
    }

    // 5. Check collision
    let collided = false;
    for (let i = 0; i < this.obstacles.length; i++) {
      if (this.checkCollision(this.obstacles[i])) {
        collided = true;
        break;
      }
    }

    if (collided) {
      this.isDead = true;
      const activeStateIdx = this.pendingJump
        ? this.pendingJump.stateIdx
        : this.pendingRun
          ? this.pendingRun.stateIdx
          : this.getCurrentStateIndex();
      const activeAction = this.pendingJump ? ACTION_JUMP : ACTION_RUN;

      if (this.mode === 'train') {
        const updateRec = this.agent.update(
          activeStateIdx,
          activeAction,
          -100.0,
          activeStateIdx,
          true
        );
        if (this.callbacks.onTransition) {
          this.callbacks.onTransition(updateRec);
        }
        transitionOccurred = true;
      }

      if (recordVisuals) {
        this.addFloatingReward(-100, this.dino.x + 12, this.dino.y - 10);
      }

      this.pendingJump = null;
      this.pendingRun = null;

      const finalScore = Math.floor(this.score);
      this.history.push({
        episode: this.episode,
        score: finalScore,
        epsilon: this.agent.epsilon,
      });
      if (this.history.length > 200) {
        this.history.shift();
      }

      episodeEnded = true;
      if (this.callbacks.onEpisodeEnd) {
        this.callbacks.onEpisodeEnd(finalScore, this.episode);
      }

      return { transitionOccurred, episodeEnded };
    }

    // 6. Check landing completion of a pending JUMP
    if (this.dino.isGrounded && this.pendingJump) {
      const cleared = this.pendingJump.clearedObstacles;
      const reward = cleared > 0 ? 20.0 * cleared : -18.0;
      const nextStateIdx = this.getCurrentStateIndex();

      if (this.mode === 'train') {
        const updateRec = this.agent.update(
          this.pendingJump.stateIdx,
          ACTION_JUMP,
          reward,
          nextStateIdx,
          false
        );
        if (this.callbacks.onTransition) {
          this.callbacks.onTransition(updateRec);
        }
        transitionOccurred = true;
      }

      if (recordVisuals) {
        this.addFloatingReward(reward, this.dino.x + 12, this.dino.y - 12);
      }
      this.pendingJump = null;
    } else if (this.pendingRun) {
      this.pendingRun.frames += 1;
    }

    return { transitionOccurred, episodeEnded };
  }

  runHeadlessEpisodes(count = 25) {
    const prevMode = this.mode;
    this.mode = 'train';
    let completed = 0;
    let safety = 0;
    while (completed < count && safety < count * 8000) {
      safety += 1;
      const res = this.stepFrame(false);
      if (res.episodeEnded) {
        completed += 1;
      }
      if (this.isDead) {
        // Start next episode cleanly
        this.stepFrame(false);
      }
    }
    this.mode = prevMode;
  }

  render() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, LOGICAL_WIDTH, LOGICAL_HEIGHT);

    // Background sky (clean aqua-ivory)
    const skyGrad = ctx.createLinearGradient(0, 0, 0, GROUND_Y);
    skyGrad.addColorStop(0, '#e3f4ef');
    skyGrad.addColorStop(1, '#f8fdfa');
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, LOGICAL_WIDTH, GROUND_Y);

    // Subtle clouds
    ctx.fillStyle = '#c2e3da';
    for (const c of this.clouds) {
      ctx.fillRect(c.x, c.y, c.w, 8);
      ctx.fillRect(c.x + 10, c.y - 6, c.w - 20, 6);
    }

    // Distance Zones Overlay ahead of Dino
    this.renderDistanceZones(ctx);

    // Ground track & pebbles (stronger earth-aqua separation)
    ctx.fillStyle = '#bfe4da';
    ctx.fillRect(0, GROUND_Y, LOGICAL_WIDTH, LOGICAL_HEIGHT - GROUND_Y);

    ctx.strokeStyle = '#145348';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, GROUND_Y);
    ctx.lineTo(LOGICAL_WIDTH, GROUND_Y);
    ctx.stroke();

    ctx.fillStyle = '#68a699';
    for (const p of this.groundPebbles) {
      ctx.fillRect(p.x, GROUND_Y + p.yOffset, p.w, 2);
    }

    // Draw obstacles (cacti)
    for (const obs of this.obstacles) {
      this.drawCactus(ctx, obs);
    }

    // Draw sensor caliper line from Dino front to next obstacle
    this.renderSensorCaliper(ctx);

    // Draw Dino
    this.drawDino(ctx);

    // Draw Decision tag above Dino
    this.renderDecisionTag(ctx);

    // Floating rewards
    for (const f of this.floatingRewards) {
      const alpha = Math.max(0, f.life / f.maxLife);
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.font = '600 13px "IBM Plex Mono", monospace';
      ctx.fillStyle = f.value > 0 ? '#059669' : f.value > -50 ? '#d97706' : '#dc2626';
      ctx.fillText(f.text, f.x, f.y);
      ctx.restore();
    }
  }

  renderDistanceZones(ctx) {
    const originX = this.dino.x + this.dino.w;
    const dist = this.getDistanceToNextObstacle();
    const activeDistBin = DIST_BINS.findIndex((b) => dist <= b.max);

    const boundaries = [0, 45, 95, 145, 210, 300, 460];
    ctx.save();
    for (let i = 0; i < DIST_BINS.length; i++) {
      const xStart = originX + boundaries[i];
      const xEnd = Math.min(LOGICAL_WIDTH, originX + boundaries[i + 1]);
      if (xStart >= LOGICAL_WIDTH) break;

      const isActive = i === activeDistBin;
      if (isActive) {
        ctx.fillStyle = 'rgba(29, 78, 216, 0.11)';
        ctx.fillRect(xStart, 22, xEnd - xStart, GROUND_Y - 22);
      }

      // Vertical zone separator
      ctx.strokeStyle = isActive ? 'rgba(29, 78, 216, 0.55)' : '#9fd1c5';
      ctx.lineWidth = isActive ? 1.5 : 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xStart, 24);
      ctx.lineTo(xStart, GROUND_Y);
      ctx.stroke();

      // Zone code at top
      ctx.setLineDash([]);
      ctx.font = isActive
        ? '600 11px "IBM Plex Mono", monospace'
        : '500 10px "IBM Plex Mono", monospace';
      ctx.fillStyle = isActive ? '#1d4ed8' : '#3d6b63';
      const labelX = xStart + 6;
      if (labelX + 18 < xEnd) {
        ctx.fillText(DIST_BINS[i].short, labelX, 36);
      }
    }
    ctx.restore();
  }

  renderSensorCaliper(ctx) {
    const nextObs = this.getNextObstacle();
    if (!nextObs) return;
    const startX = this.dino.x + this.dino.w;
    const endX = nextObs.x;
    if (endX <= startX) return;

    const lineY = GROUND_Y + 22;
    const dist = Math.round(endX - startX);
    const state = STATES[this.getCurrentStateIndex()];

    ctx.save();
    ctx.strokeStyle = '#1d4ed8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(startX, lineY - 5);
    ctx.lineTo(startX, lineY + 5);
    ctx.moveTo(startX, lineY);
    ctx.lineTo(endX, lineY);
    ctx.moveTo(endX, lineY - 5);
    ctx.lineTo(endX, lineY + 5);
    ctx.stroke();

    const label = `${dist} px · ${state.code}`;
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    const textW = ctx.measureText(label).width;
    const midX = Math.max(startX + 4, Math.min(LOGICAL_WIDTH - textW - 12, (startX + endX - textW) / 2));

    ctx.fillStyle = '#eff6ff';
    ctx.fillRect(midX - 5, lineY - 8, textW + 10, 16);
    ctx.fillStyle = '#1e3a8a';
    ctx.fillText(label, midX, lineY + 4);
    ctx.restore();
  }

  renderDecisionTag(ctx) {
    const dec = this.lastDecision;
    if (!dec) return;

    let label;
    let color;
    if (this.isDead) {
      label = '✕ COLISÃO';
      color = '#dc2626';
    } else if (dec.isExploration) {
      label = `⚡ ${ACTION_LABELS[dec.action].toUpperCase()} (ε)`;
      color = '#b45309';
    } else if (dec.action === ACTION_JUMP) {
      label = '▲ SALTAR';
      color = '#1d4ed8';
    } else {
      label = '→ CORRER';
      color = '#047857';
    }

    ctx.save();
    ctx.font = '600 11px "IBM Plex Mono", monospace';
    const x = Math.max(12, this.dino.x - 4);
    const y = Math.max(16, this.dino.y - 10);
    ctx.fillStyle = color;
    ctx.fillText(label, x, y);
    ctx.restore();
  }

  drawDino(ctx) {
    const { x, y, legFrame, isGrounded } = this.dino;
    ctx.save();
    ctx.fillStyle = this.isDead ? '#dc2626' : '#072621';

    // Head & snout
    ctx.fillRect(x + 14, y, 20, 14);
    ctx.fillRect(x + 20, y + 14, 10, 3);
    // Eye
    ctx.fillStyle = '#f8fdfa';
    ctx.fillRect(x + 20, y + 4, 4, 4);
    if (!this.isDead) {
      ctx.fillStyle = '#2dd4bf';
      ctx.fillRect(x + 22, y + 5, 2, 2);
    }

    // Neck & torso
    ctx.fillStyle = this.isDead ? '#dc2626' : '#072621';
    ctx.fillRect(x + 12, y + 14, 12, 6);
    ctx.fillRect(x + 6, y + 18, 18, 16);
    // Tail
    ctx.fillRect(x, y + 20, 6, 8);
    ctx.fillRect(x + 2, y + 16, 4, 6);
    // Tiny arm
    ctx.fillRect(x + 24, y + 22, 6, 3);
    ctx.fillRect(x + 28, y + 24, 2, 3);

    // Legs
    if (!isGrounded) {
      // Tucked jump pose
      ctx.fillRect(x + 9, y + 34, 5, 7);
      ctx.fillRect(x + 17, y + 32, 5, 7);
    } else if (legFrame === 0) {
      ctx.fillRect(x + 8, y + 34, 5, 10);
      ctx.fillRect(x + 10, y + 42, 6, 2);
      ctx.fillRect(x + 17, y + 34, 5, 6);
    } else {
      ctx.fillRect(x + 8, y + 34, 5, 6);
      ctx.fillRect(x + 17, y + 34, 5, 10);
      ctx.fillRect(x + 19, y + 42, 6, 2);
    }
    ctx.restore();
  }

  drawCactus(ctx, obs) {
    const { x, y, w, h, type } = obs;
    ctx.save();
    ctx.fillStyle = '#047857';

    if (type === 'double') {
      // Two side-by-side stems
      ctx.fillRect(x + 5, y + 4, 10, h - 4);
      ctx.fillRect(x + 7, y, 6, 4);
      ctx.fillRect(x, y + 12, 5, 10);
      ctx.fillRect(x, y + 10, 4, 4);

      ctx.fillRect(x + 24, y + 8, 10, h - 8);
      ctx.fillRect(x + 26, y + 4, 6, 4);
      ctx.fillRect(x + 34, y + 14, 7, 10);
      ctx.fillRect(x + 37, y + 12, 4, 4);
    } else {
      const stemW = 10;
      const stemX = x + Math.floor((w - stemW) / 2);
      ctx.fillRect(stemX, y + 3, stemW, h - 3);
      ctx.fillRect(stemX + 2, y, stemW - 4, 4);

      // Left arm
      ctx.fillRect(x, y + 12, stemX - x, 6);
      ctx.fillRect(x, y + 8, 5, 10);
      // Right arm
      ctx.fillRect(stemX + stemW, y + 16, x + w - (stemX + stemW), 6);
      ctx.fillRect(x + w - 5, y + 11, 5, 11);
    }

    ctx.restore();
  }
}
