'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const xs = Array.from({length: 241}, (_, i) => i / 40);
  const defaults = () => Object.fromEntries(Object.entries(GP.parameters).map(([key, p]) => [key, p.value]));
  const state = {
    configs: [{type: 'rbf', ...defaults()}, {type: 'periodic', ...defaults()}],
    mode: 'single', count: 8, noise: .15, selected: [80, 160],
    prior: null, posterior: null, target: null, indices: [], errors: [], values: [],
    priorSamples: [], posteriorSamples: [], running: false, yLimit: 3
  };
  const colors = ['#368a40', '#d7832d', '#8d66b0'];
  const pointColors = ['#078580', '#b058a2'];
  const indicatorValues = [0, 0];
  const chartGeometry = new Map();
  const NS = 'http://www.w3.org/2000/svg';
  let timer = null;

  function element(tag, attrs = {}, content) {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (content !== undefined) node.textContent = content;
    return node;
  }
  function add(parent, tag, attrs, content) {
    const node = element(tag, attrs, content); parent.appendChild(node); return node;
  }
  function path(parent, points, attrs = {}) {
    return add(parent, 'path', {d: points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(3)},${y.toFixed(3)}`).join(' '), fill: 'none', ...attrs});
  }
  const fmt = x => Math.abs(x) < 1e-10 ? '0' : Number(x.toPrecision(3)).toString();
  function ticks(low, high) {
    const raw = (high - low) / 5, base = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].find(v => v * base >= raw) * base;
    const result = [];
    for (let x = Math.ceil(low / step) * step; x <= high + step * 1e-8; x += step) result.push(x);
    return result;
  }
  function chart(id, joint = false) {
    const svg = $(id), width = Math.max(100, svg.clientWidth || (joint ? 180 : 760));
    const height = svg.clientHeight || 160;
    const compact = window.innerWidth >= 900;
    const bottomAxis = joint || !compact || id === 'posterior-plot';
    svg.replaceChildren(); svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    const margin = {left: joint ? 32 : 48, right: joint ? 30 : 40, top: joint ? 48 : 24, bottom: bottomAxis ? 31 : 6};
    let plotWidth = width - margin.left - margin.right, plotHeight = height - margin.top - margin.bottom;
    if (joint) {
      const side = Math.min(plotWidth, plotHeight);
      margin.left += (plotWidth - side) / 2;
      margin.top += (plotHeight - side) / 2;
      plotWidth = plotHeight = side;
    }
    const limit = state.yLimit, xMin = joint ? -limit : 0, xMax = joint ? limit : 6;
    const X = x => margin.left + (x - xMin) / (xMax - xMin) * plotWidth;
    const Y = y => margin.top + (limit - y) / (2 * limit) * plotHeight;
    for (const value of ticks(xMin, xMax)) {
      add(svg, 'line', {x1: X(value), y1: Y(-limit), x2: X(value), y2: Y(limit), stroke: '#e7edf2'});
      if (bottomAxis) add(svg, 'text', {x: X(value), y: Y(-limit) + 16, 'text-anchor': 'middle'}, fmt(value));
    }
    for (const value of ticks(-limit, limit)) {
      add(svg, 'line', {x1: X(xMin), y1: Y(value), x2: X(xMax), y2: Y(value), stroke: '#e7edf2'});
      add(svg, 'text', {x: X(xMin) - 8, y: Y(value) + 5, 'text-anchor': 'end'}, fmt(value));
    }
    add(svg, 'rect', {x: X(xMin), y: Y(limit), width: plotWidth, height: plotHeight, fill: 'none', stroke: '#cbd7e1'});
    if (bottomAxis) add(svg, 'text', {
      x: X(xMax), y: height - 1, 'text-anchor': 'end',
      ...(joint ? {style: `fill:${pointColors[0]}`} : {})
    }, joint ? 'f(x₁)' : '入力 x');
    if (joint) {
      const title = add(svg, 'text', {x: X(0), y: 15, 'text-anchor': 'middle', class: 'joint-title'});
      add(title, 'tspan', {}, 'p(f(x');
      add(title, 'tspan', {dy: 4, 'font-size': 10, 'font-style': 'normal'}, '1');
      add(title, 'tspan', {dy: -4}, '), f(x');
      add(title, 'tspan', {dy: 4, 'font-size': 10, 'font-style': 'normal'}, '2');
      add(title, 'tspan', {dy: -4}, '))');
      add(svg, 'text', {x: 2, y: 31, style: `fill:${pointColors[1]}`}, 'f(x₂)');
    }
    const clipId = `${id}-clip`;
    const clip = add(add(svg, 'defs'), 'clipPath', {id: clipId});
    add(clip, 'rect', {x: X(xMin), y: Y(limit), width: plotWidth, height: plotHeight});
    const layer = add(svg, 'g', {'clip-path': `url(#${clipId})`});
    const geometry = {svg, layer, X, Y, limit, joint, plotWidth, plotHeight};
    chartGeometry.set(id, geometry);
    return geometry;
  }
  // The value axis uses the same scale as the function/joint plot. Density
  // heights are normalized for visibility, so even narrow posteriors fit.
  function marginal(parent, mean, variance, project, color, amplitude, index) {
    const sigma = Math.sqrt(Math.max(0, variance));
    const attrs = {stroke: color, 'stroke-width': 1.8, 'data-kind': 'marginal', 'data-point': index};
    if (sigma <= 1e-7 * state.yLimit) {
      const line = path(parent, [project(mean, 0), project(mean, amplitude)], attrs);
      add(line, 'title', {}, '分散がほぼ0の分布');
      return;
    }
    const points = Array.from({length: 101}, (_, n) => {
      const z = -4 + 8 * n / 100;
      return project(mean + z * sigma, amplitude * Math.exp(-.5 * z * z));
    });
    path(parent, [project(mean - 4 * sigma, 0), ...points, project(mean + 4 * sigma, 0)], {
      ...attrs, fill: color, 'fill-opacity': .09
    });
  }
  function marginalLayer(c, id, side) {
    const {svg, X, Y, limit, joint} = c;
    const clipId = `${id}-marginal-clip`;
    const clip = add(svg.querySelector('defs'), 'clipPath', {id: clipId});
    // Keep value tails within the visible axes, but allow density curves to
    // protrude beyond the top/right boundary of the joint distribution.
    if (joint) {
      if (side === 0) add(clip, 'rect', {x: X(-limit), y: Y(limit) - 24, width: X(limit) - X(-limit), height: 24});
      else add(clip, 'rect', {x: X(limit), y: Y(limit), width: 24, height: Y(-limit) - Y(limit)});
    } else add(clip, 'rect', {x: X(0), y: Y(limit), width: X(6) - X(0) + 34, height: Y(-limit) - Y(limit)});
    return add(svg, 'g', {'clip-path': `url(#${clipId})`, 'pointer-events': 'none'});
  }
  function addIndicators(c) {
    pointColors.forEach((color, j) => add(c.svg, 'line', {
      'data-kind': 'value-indicator', 'data-point': j,
      stroke: color, 'stroke-width': 2.5, 'pointer-events': 'none'
    }));
  }
  function updateIndicators() {
    for (const [id, c] of chartGeometry) {
      if (id === 'target-plot') continue;
      const {svg, X, Y, limit, joint} = c;
      svg.querySelectorAll('[data-kind="value-indicator"]').forEach((bar, j) => {
        const value = Math.max(-limit, Math.min(limit, indicatorValues[j]));
        let coordinates;
        if (joint && j === 0) coordinates = {x1: X(value), x2: X(value), y1: Y(limit) - 5, y2: Y(limit) + 5};
        else if (joint) coordinates = {x1: X(limit) - 5, x2: X(limit) + 5, y1: Y(value), y2: Y(value)};
        else coordinates = {x1: X(xs[state.selected[j]]) - 5, x2: X(xs[state.selected[j]]) + 5, y1: Y(value), y2: Y(value)};
        for (const [key, coordinate] of Object.entries(coordinates)) bar.setAttribute(key, coordinate);
      });
    }
  }
  function observations(chart) {
    state.indices.forEach((i, j) => add(chart.layer, 'circle', {cx: chart.X(xs[i]), cy: chart.Y(state.values[j]), r: 4.1, fill: '#d62728', stroke: '#fff', 'stroke-width': 1.3}));
  }
  function drawFunction(id, distribution, samples, showData = false) {
    const c = chart(id), {X, Y, layer, svg, limit} = c;
    if (distribution) {
      const {mean, cov} = distribution;
      const upper = xs.map((x, i) => [X(x), Y(mean[i] + 1.96 * Math.sqrt(cov[i][i]))]);
      const lower = xs.map((x, i) => [X(x), Y(mean[i] - 1.96 * Math.sqrt(cov[i][i]))]).reverse();
      const band = path(layer, upper.concat(lower), {fill: '#e1edf6', 'fill-opacity': .85});
      band.setAttribute('d', band.getAttribute('d') + ' Z');
      path(layer, xs.map((x, i) => [X(x), Y(mean[i])]), {stroke: '#2460c7', 'stroke-width': 1.8, 'data-kind': 'mean'});
    }
    samples.forEach((sample, j) => path(layer, xs.map((x, i) => [X(x), Y(sample[i])]), {
      stroke: distribution ? colors[j] : '#000', 'stroke-width': distribution ? 1.7 : 2.2,
      'stroke-dasharray': distribution ? '7 3 1 3' : '1 5', 'stroke-linecap': 'round',
      'data-kind': distribution ? 'sample' : 'target'
    }));
    if (distribution) state.selected.forEach((i, j) => {
      const color = pointColors[j];
      add(layer, 'line', {x1: X(xs[i]), x2: X(xs[i]), y1: Y(-limit), y2: Y(limit), stroke: color, 'stroke-dasharray': '4 4', 'stroke-width': 1.3});
      add(svg, 'text', {x: Math.max(X(0) + 15, Math.min(X(6) - 15, X(xs[i]))), y: 16, 'text-anchor': 'middle', style: `fill:${color}`}, j ? 'x₂' : 'x₁');
      samples.forEach((sample, k) => add(layer, 'circle', {cx: X(xs[i]), cy: Y(sample[i]), r: 3.2, fill: colors[k], stroke: '#fff', 'stroke-width': 1}));
    });
    if (distribution) {
      const marginals = marginalLayer(c, id);
      state.selected.forEach((i, j) => marginal(marginals, distribution.mean[i], distribution.cov[i][i],
        (value, density) => [X(xs[i]) + density, Y(value)], pointColors[j], 32, j));
      addIndicators(c);
    }
    if (showData) observations(c);
  }
  function drawJoint(id, distribution, samples, infoId) {
    const {mean, cov} = distribution, [i, j] = state.selected;
    const c = chart(id, true), {layer, X, Y, limit} = c, a = cov[i][i], b = cov[i][j], d = cov[j][j];
    for (const [mass, opacity] of [[.95, .12], [.8, .16], [.5, .22]]) {
      const {major, minor, angle} = GP.ellipse(a, b, d, mass);
      const points = Array.from({length: 101}, (_, n) => {
        const t = n * 2 * Math.PI / 100;
        return [X(mean[i] + major * Math.cos(t) * Math.cos(angle) - minor * Math.sin(t) * Math.sin(angle)), Y(mean[j] + major * Math.cos(t) * Math.sin(angle) + minor * Math.sin(t) * Math.cos(angle))];
      });
      path(layer, points, {fill: '#5590c3', 'fill-opacity': opacity, stroke: '#4d85b4', 'stroke-width': 1, 'data-kind': 'contour'});
    }
    add(layer, 'circle', {cx: X(mean[i]), cy: Y(mean[j]), r: 2.5, fill: '#2460c7'});
    samples.forEach((sample, n) => add(layer, 'circle', {cx: X(sample[i]), cy: Y(sample[j]), r: 4, fill: colors[n], stroke: '#fff', 'stroke-width': 1.2}));
    marginal(marginalLayer(c, `${id}-top`, 0), mean[i], a, (value, density) => [X(value), Y(limit) - density], pointColors[0], 21, 0);
    marginal(marginalLayer(c, `${id}-right`, 1), mean[j], d, (value, density) => [X(limit) + density, Y(value)], pointColors[1], 21, 1);
    addIndicators(c);
    const rho = a * d > 1e-16 ? fmt(Math.max(-1, Math.min(1, b / Math.sqrt(a * d)))) : '未定義（分散0）';
    // Native MathML keeps the matrix in display style without a CDN dependency.
    const info = $(infoId);
    info.innerHTML = `<span>共分散行列</span><math display="block" aria-label="共分散行列"><mi>Σ</mi><mo>=</mo><mrow><mo stretchy="true">[</mo><mtable columnspacing="0.8em" rowspacing="0.35em"><mtr><mtd><mn>${fmt(a)}</mn></mtd><mtd><mn>${fmt(b)}</mn></mtd></mtr><mtr><mtd><mn>${fmt(b)}</mn></mtd><mtd><mn>${fmt(d)}</mn></mtd></mtr></mtable><mo stretchy="true">]</mo></mrow></math><span>相関係数：${rho}</span>`;
  }
  function updateLimit(reset = false) {
    let limit = 1;
    for (const dist of [state.prior, state.posterior]) for (let i = 0; i < xs.length; i++) limit = Math.max(limit, Math.abs(dist.mean[i]) + 2.6 * Math.sqrt(dist.cov[i][i]));
    for (const values of [state.target, state.values, ...state.priorSamples, ...state.posteriorSamples]) for (const value of values) limit = Math.max(limit, Math.abs(value) * 1.08);
    state.yLimit = reset ? limit : Math.max(limit, state.yLimit);
  }
  function draw() {
    if (!state.prior || !state.posterior) return;
    const panel = document.querySelector('.visualization');
    // Reserve space for the legend; adapt to the browser's actual content height.
    if (window.innerWidth >= 900) {
      const top = panel.getBoundingClientRect().top + window.scrollY;
      const rowHeight = Math.max(110, Math.min(240, Math.floor((window.innerHeight - top - 44) / 3)));
      panel.style.setProperty('--row-height', `${rowHeight}px`);
    } else panel.style.removeProperty('--row-height');
    drawFunction('prior-plot', state.prior, state.priorSamples);
    drawFunction('target-plot', null, [state.target], true);
    drawFunction('posterior-plot', state.posterior, state.posteriorSamples, true);
    drawJoint('prior-joint', state.prior, state.priorSamples, 'prior-cov');
    drawJoint('posterior-joint', state.posterior, state.posteriorSamples, 'posterior-cov');
    updateIndicators();
  }
  function samplePaths() {
    state.priorSamples = Array.from({length: 3}, () => GP.sample(state.prior.mean, state.prior.factor));
    state.posteriorSamples = Array.from({length: 3}, () => GP.sample(state.posterior.mean, state.posterior.factor));
  }
  function condition() {
    state.values = state.indices.map((i, j) => state.target[i] + state.noise * state.errors[j]);
    state.posterior = GP.posterior(state.prior.cov, state.indices, state.values, state.noise);
    // Subtraction in a nearly deterministic posterior inherits prior-scale roundoff.
    state.posterior.factor = GP.cholesky(state.posterior.cov, Math.max(...state.prior.cov.map((row, i) => row[i])));
    samplePaths(); updateLimit(true); draw();
  }
  function generateData() {
    state.indices = GP.randomIndices(xs.length, state.count);
    state.errors = state.indices.map(() => GP.normal());
    condition();
  }
  function generateTarget() {
    state.target = GP.sample(state.prior.mean, state.prior.factor);
    generateData();
  }
  function rebuild() {
    const k = GP.combine(GP.kernel(state.configs[0]), GP.kernel(state.configs[1]), state.mode);
    const cov = GP.covariance(xs, k);
    state.prior = {mean: new Float64Array(xs.length), cov, factor: GP.cholesky(cov)};
    generateTarget();
  }
  function stopAnimation() {
    state.running = false; clearInterval(timer); timer = null;
    $('animate').textContent = '自動再生成を開始'; $('animate').setAttribute('aria-pressed', 'false');
  }
  function act(action, message) {
    try {
      action(); $('status').classList.remove('error'); $('status').textContent = message;
    } catch (error) {
      stopAnimation(); $('status').classList.add('error');
      $('status').textContent = `計算エラー：${error.message}`;
      console.error(error);
    }
  }
  function renderKernels() {
    const container = $('kernels'); container.replaceChildren();
    container.classList.toggle('dual', state.mode !== 'single');
    state.configs.slice(0, state.mode === 'single' ? 1 : 2).forEach((config, i) => {
      const block = document.createElement('fieldset'); block.className = 'kernel-block';
      const legend = document.createElement('legend'); legend.textContent = `カーネル k${i + 1}`; block.appendChild(legend);
      const select = document.createElement('select'); select.id = `kernel-${i}`; select.setAttribute('aria-label', `カーネル k${i + 1} の種類`);
      for (const [key, def] of Object.entries(GP.definitions)) {
        const option = document.createElement('option'); option.value = key; option.textContent = def.name; select.appendChild(option);
      }
      select.value = config.type; block.appendChild(select);
      select.title = GP.definitions[config.type].formula;
      select.addEventListener('change', () => {
        config.type = select.value; renderKernels(); act(rebuild, 'カーネルを変更し、ターゲット関数と訓練データを更新しました。');
      });
      const formula = document.createElement('p'); formula.className = 'kernel-formula'; formula.textContent = GP.definitions[config.type].formula; block.appendChild(formula);
      const params = document.createElement('div'); params.className = 'parameters'; block.appendChild(params);
      for (const key of GP.definitions[config.type].parameters) {
        const p = GP.parameters[key], wrapper = document.createElement('div'); wrapper.className = 'parameter';
        const label = document.createElement('label'), input = document.createElement('input'), output = document.createElement('output');
        input.id = `kernel-${i}-${key}`; input.type = 'range'; input.min = p.min; input.max = p.max; input.step = p.step; input.value = config[key];
        label.htmlFor = input.id; label.textContent = p.label; output.htmlFor = input.id; output.textContent = config[key].toFixed(2);
        label.appendChild(output); wrapper.append(label, input); params.appendChild(wrapper);
        input.addEventListener('input', () => {output.textContent = Number(input.value).toFixed(2);});
        input.addEventListener('change', () => {config[key] = Number(input.value); act(rebuild, 'カーネルの設定を変更し、ターゲット関数と訓練データを更新しました。');});
      }
      container.appendChild(block);
    });
  }
  $('combination').addEventListener('change', event => {state.mode = event.target.value; renderKernels(); act(rebuild, 'カーネルの組み合わせを変更し、ターゲット関数と訓練データを更新しました。');});
  ['x1', 'x2'].forEach((id, j) => $(id).addEventListener('input', () => {
    state.selected[j] = Number($(id).value); $(`${id}-value`).textContent = xs[state.selected[j]].toFixed(2); draw();
  }));
  $('data-count').addEventListener('input', () => {$('count-value').textContent = $('data-count').value;});
  $('data-count').addEventListener('change', () => {state.count = Number($('data-count').value); act(generateData, '指定した個数の訓練データを生成しました。');});
  $('noise').addEventListener('input', () => {$('noise-value').textContent = Number($('noise').value).toFixed(2);});
  $('noise').addEventListener('change', () => {state.noise = Number($('noise').value); act(condition, '観測ノイズの大きさと事後分布を更新しました。');});
  $('new-data').addEventListener('click', () => act(generateData, 'ターゲット関数を固定し、訓練データを生成し直しました。'));
  $('new-target').addEventListener('click', () => act(generateTarget, 'ターゲット関数と訓練データを生成し直しました。'));
  const resample = () => {samplePaths(); updateLimit(); draw();};
  $('resample').addEventListener('click', () => act(resample, '事前分布・事後分布から、それぞれ3本を新しく生成しました。'));
  $('animate').addEventListener('click', () => {
    if (state.running) {stopAnimation(); $('status').textContent = '自動再生成を停止しました。'; return;}
    state.running = true; $('animate').textContent = '自動再生成を停止'; $('animate').setAttribute('aria-pressed', 'true');
    $('status').textContent = '1.5秒ごとにサンプルパスを引き直しています。ターゲット関数と訓練データは固定です。';
    timer = setInterval(() => {if (!document.hidden) act(resample, '自動再生成中：ターゲット関数と訓練データは固定です。');}, 1500);
  });
  window.addEventListener('pagehide', stopAnimation);
  ['prior-plot', 'posterior-plot', 'prior-joint', 'posterior-joint'].forEach(id => {
    $(id).addEventListener('pointermove', event => {
      const c = chartGeometry.get(id);
      if (!c) return;
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(c.svg.getScreenCTM().inverse());
      const clamp = value => Math.max(-c.limit, Math.min(c.limit, value));
      const verticalValue = clamp(c.limit - (point.y - c.Y(c.limit)) / c.plotHeight * 2 * c.limit);
      indicatorValues[0] = c.joint ? clamp(-c.limit + (point.x - c.X(-c.limit)) / c.plotWidth * 2 * c.limit) : verticalValue;
      indicatorValues[1] = verticalValue;
      updateIndicators();
    });
  });
  let resizeFrame;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(draw);
  });
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(draw);
  }).observe(document.querySelector('main'));
  renderKernels(); act(rebuild, 'カーネルや2点の位置を変えて、事前分布と事後分布を比較できます。');
})();
