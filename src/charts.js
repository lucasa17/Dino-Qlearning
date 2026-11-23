const CHART_W = 800;
const CHART_H = 190;

export class LearningChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.configureDpi();
  }

  configureDpi() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = CHART_W * dpr;
    this.canvas.height = CHART_H * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  render(history) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, CHART_W, CHART_H);

    const padL = 46;
    const padR = 44;
    const padT = 16;
    const padB = 26;
    const plotW = CHART_W - padL - padR;
    const plotH = CHART_H - padT - padB;

    // Background & grid
    ctx.fillStyle = '#fcfffe';
    ctx.fillRect(0, 0, CHART_W, CHART_H);

    ctx.strokeStyle = '#d3ede6';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#4e8a7f';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    if (!history || history.length === 0) {
      ctx.fillStyle = '#49756d';
      ctx.font = '500 13px "Plus Jakarta Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        'Conclua o primeiro episódio para visualizar a curva de aprendizado.',
        CHART_W / 2,
        CHART_H / 2 + 4
      );
      ctx.textAlign = 'left';
      return;
    }

    const maxScore = Math.max(100, ...history.map((h) => h.score));
    const n = history.length;

    // Y-axis ticks (Score)
    ctx.fillStyle = '#49756d';
    ctx.font = '500 10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = Math.round((maxScore * (4 - i)) / 4);
      const y = padT + (plotH / 4) * i + 3;
      ctx.fillText(String(val), padL - 6, y);
    }

    // Right Y-axis ticks (Epsilon 0..1)
    ctx.textAlign = 'left';
    ctx.fillStyle = '#d97706';
    ctx.fillText('ε 1.0', padL + plotW + 6, padT + 4);
    ctx.fillText('ε 0.0', padL + plotW + 6, padT + plotH);

    const getX = (idx) => (n === 1 ? padL + plotW / 2 : padL + (idx / (n - 1)) * plotW);
    const getYScore = (score) => padT + plotH - (score / maxScore) * plotH;
    const getYEps = (eps) => padT + plotH - eps * plotH;

    // Raw episode scores line
    ctx.strokeStyle = '#6da397';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = getX(i);
      const y = getYScore(history[i].score);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Epsilon decay dashed line
    ctx.save();
    ctx.strokeStyle = '#d97706';
    ctx.lineWidth = 1.75;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = getX(i);
      const y = getYEps(history[i].epsilon);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 10-episode moving average line
    ctx.strokeStyle = '#1d4ed8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const start = Math.max(0, i - 9);
      let sum = 0;
      for (let j = start; j <= i; j++) sum += history[j].score;
      const avg = sum / (i - start + 1);
      const x = getX(i);
      const y = getYScore(avg);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis episode labels
    ctx.fillStyle = '#49756d';
    ctx.textAlign = 'center';
    ctx.fillText(`Ep. ${history[0].episode}`, padL + 16, CHART_H - 7);
    if (n > 1) {
      ctx.fillText(`Ep. ${history[n - 1].episode}`, padL + plotW - 16, CHART_H - 7);
    }
    ctx.textAlign = 'left';
  }
}
