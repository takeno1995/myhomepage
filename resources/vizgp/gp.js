/* Gaussian-process numerics. All arrays are local to the visitor's browser. */
'use strict';
globalThis.GP = (() => {
  const definitions = {
    rbf: {name: '二乗指数（RBF）', formula: 'σ² exp(−(x − x′)² / (2ℓ²))', parameters: ['variance', 'length']},
    matern12: {name: '指数（Matérn 1/2）', formula: 'σ² exp(−|x − x′| / ℓ)', parameters: ['variance', 'length']},
    matern32: {name: 'Matérn 3/2', formula: 'σ² (1 + √3r) exp(−√3r),  r = |x − x′| / ℓ', parameters: ['variance', 'length']},
    matern52: {name: 'Matérn 5/2', formula: 'σ² (1 + √5r + 5r²/3) exp(−√5r),  r = |x − x′| / ℓ', parameters: ['variance', 'length']},
    periodic: {name: '周期', formula: 'σ² exp(−2 sin²(π|x − x′| / p) / ℓ²)', parameters: ['variance', 'length', 'period']},
    linear: {name: '線形', formula: 'σ² (x − x꜀)(x′ − x꜀) + σᵦ²', parameters: ['variance', 'bias', 'center']}
  };
  const parameters = {
    variance: {label: '分散 σ²', min: 0, max: 4, step: .01, value: 1},
    length: {label: '長さ尺度 ℓ', min: .05, max: 10, step: .01, value: 1},
    period: {label: '周期 p', min: .1, max: 10, step: .01, value: 2},
    bias: {label: '切片の分散 σᵦ²', min: 0, max: 4, step: .01, value: 0},
    center: {label: '中心 x꜀', min: -2, max: 8, step: .1, value: 2}
  };
  function kernel(config) {
    const {type, variance: v, length: l, period: p, bias: b, center: c} = config;
    return (x, y) => {
      const r = Math.abs(x - y) / l;
      switch (type) {
        case 'rbf': return v * Math.exp(-.5 * r * r);
        case 'matern12': return v * Math.exp(-r);
        case 'matern32': {const z = Math.sqrt(3) * r; return v * (1 + z) * Math.exp(-z);}
        case 'matern52': {const z = Math.sqrt(5) * r; return v * (1 + z + z * z / 3) * Math.exp(-z);}
        case 'periodic': return v * Math.exp(-2 * (Math.sin(Math.PI * (x - y) / p) / l) ** 2);
        case 'linear': return v * (x - c) * (y - c) + b;
        default: throw new Error('未知のカーネルです。');
      }
    };
  }
  function combine(a, b, mode) {
    if (mode === 'sum') return (x, y) => a(x, y) + b(x, y);
    if (mode === 'product') return (x, y) => a(x, y) * b(x, y);
    return a;
  }
  const matrix = n => Array.from({length: n}, () => new Float64Array(n));
  function covariance(xs, k) {
    const c = matrix(xs.length);
    for (let i = 0; i < xs.length; i++) for (let j = 0; j <= i; j++) c[i][j] = c[j][i] = k(xs[i], xs[j]);
    return c;
  }
  // Relative diagonal regularization handles low-rank linear/periodic kernels.
  // An identically zero covariance remains exactly zero.
  function cholesky(a, referenceScale = 0) {
    const n = a.length, scale = Math.max(referenceScale, ...a.map((row, i) => row[i]), 0);
    if (!scale) return matrix(n);
    for (const relative of [1e-12, 1e-10, 1e-8, 1e-6]) {
      const result = matrix(n), jitter = scale * relative;
      let valid = true;
      for (let i = 0; i < n && valid; i++) {
        for (let j = 0; j <= i; j++) {
          let s = a[i][j] + (i === j ? jitter : 0);
          for (let k = 0; k < j; k++) s -= result[i][k] * result[j][k];
          if (i === j) {
            if (!(s > 0)) {valid = false; break;}
            result[i][j] = Math.sqrt(s);
          } else result[i][j] = s / result[j][j];
        }
      }
      if (valid) return result;
    }
    throw new Error('共分散行列を分解できませんでした。カーネルの設定を変更してください。');
  }
  function forward(l, b) {
    const y = new Float64Array(b.length);
    for (let i = 0; i < b.length; i++) {
      let s = b[i];
      for (let j = 0; j < i; j++) s -= l[i][j] * y[j];
      y[i] = l[i][i] ? s / l[i][i] : 0;
    }
    return y;
  }
  function posterior(prior, indices, values, noise) {
    const n = prior.length, m = indices.length, mean = new Float64Array(n);
    if (!m) return {mean, cov: prior.map(row => row.slice())};
    const obs = indices.map((i, a) => Float64Array.from(indices, (j, b) => prior[i][j] + (a === b ? noise * noise : 0)));
    const l = cholesky(obs), whitened = forward(l, values);
    const v = prior.map(row => forward(l, Float64Array.from(indices, i => row[i])));
    const cov = matrix(n);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < m; k++) mean[i] += v[i][k] * whitened[k];
      for (let j = 0; j <= i; j++) {
        let c = prior[i][j];
        for (let k = 0; k < m; k++) c -= v[i][k] * v[j][k];
        cov[i][j] = cov[j][i] = i === j ? Math.max(0, c) : c;
      }
    }
    return {mean, cov};
  }
  function normal(random = Math.random) {
    return Math.sqrt(-2 * Math.log(Math.max(Number.MIN_VALUE, random()))) * Math.cos(2 * Math.PI * random());
  }
  function sample(mean, factor, random = Math.random) {
    const z = Float64Array.from(mean, () => normal(random));
    return Float64Array.from(mean, (mu, i) => {
      for (let j = 0; j <= i; j++) mu += factor[i][j] * z[j];
      return mu;
    });
  }
  function randomIndices(size, count, random = Math.random) {
    const pool = Array.from({length: size}, (_, i) => i);
    for (let i = 0; i < count; i++) {
      const j = i + Math.floor(random() * (size - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, count).sort((a, b) => a - b);
  }
  function ellipse(a, b, d, mass) {
    const delta = Math.hypot(a - d, 2 * b), radius = -2 * Math.log(1 - mass);
    return {major: Math.sqrt(Math.max(0, (a + d + delta) / 2) * radius), minor: Math.sqrt(Math.max(0, (a + d - delta) / 2) * radius), angle: Math.atan2(2 * b, a - d) / 2};
  }
  return {definitions, parameters, kernel, combine, covariance, cholesky, forward, posterior, normal, sample, randomIndices, ellipse};
})();
