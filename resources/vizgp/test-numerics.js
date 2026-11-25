'use strict';
globalThis.numericalResults = [];
(() => {
  function test(name, fn) {fn(); numericalResults.push(name);}
  function assert(ok, message) {if (!ok) throw new Error(message);}
  function near(a, b, tol = 1e-7) {assert(Math.abs(a - b) <= tol, `${a} != ${b}`);}
  const config = type => ({type, variance: 1, length: 1, period: 2, bias: 0, center: 2});
  const xs = Array.from({length: 61}, (_, i) => i / 10);
  const k = GP.kernel(config('rbf'));
  test('1点の観測に対する事後平均・共分散を解析式と比較', () => {
    const prior = GP.covariance([0, 1], k), post = GP.posterior(prior, [0], [2], .5);
    near(post.mean[0], 2 / 1.25); near(post.mean[1], 2 * Math.exp(-.5) / 1.25);
    near(post.cov[1][1], 1 - Math.exp(-1) / 1.25);
  });
  test('データ0件では事後分布と事前分布が一致', () => {
    const prior = GP.covariance(xs, k), post = GP.posterior(prior, [], [], 0);
    prior.forEach((row, i) => row.forEach((v, j) => near(v, post.cov[i][j], 0)));
  });
  test('ノイズなしの観測点で補間し、分散が0に近づく', () => {
    const prior = GP.covariance(xs, k), post = GP.posterior(prior, [4, 25, 57], [1, -2, .7], 0);
    [4, 25, 57].forEach((i, j) => {near(post.mean[i], [1, -2, .7][j]); near(post.cov[i][i], 0);});
  });
  test('全6カーネルと全ての和・積で分解と条件付けが有限', () => {
    const kernels = Object.keys(GP.definitions).map(type => GP.kernel(config(type)));
    for (const a of kernels) for (const b of kernels) for (const mode of ['single', 'sum', 'product']) {
      const prior = GP.covariance(xs, GP.combine(a, b, mode)), l = GP.cholesky(prior);
      const target = GP.sample(new Float64Array(xs.length), l);
      for (const noise of [0, .15]) {
        const indices = [2, 10, 20, 30, 50], post = GP.posterior(prior, indices, indices.map(i => target[i]), noise);
        const pl = GP.cholesky(post.cov, Math.max(...prior.map((row, i) => row[i])));
        GP.sample(post.mean, pl).forEach(v => assert(Number.isFinite(v), `${mode}: 非有限のサンプル`));
        post.cov.forEach((row, i) => assert(row[i] <= prior[i][i] + 1e-7, '事後分散が増加'));
      }
    }
  });
  test('分散0とノイズ0でもサンプルと平均は厳密に0', () => {
    const prior = GP.covariance(xs, () => 0), post = GP.posterior(prior, [1, 3], [0, 0], 0);
    GP.sample(post.mean, GP.cholesky(post.cov)).forEach(v => near(v, 0, 0));
  });
  test('特異な2変量分布は線または点になる', () => {
    const line = GP.ellipse(1, 1, 1, .95), point = GP.ellipse(0, 0, 0, .95);
    near(line.minor, 0); near(line.angle, Math.PI / 4); near(point.major, 0);
  });
  test('サンプルの経験平均・共分散を指定した分布と比較', () => {
    let seed = 123456;
    const rng = () => {seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return (seed + .5) / 4294967296;};
    const factor = GP.cholesky([[1, .6], [.6, 2]]), count = 20000;
    let sum = [0, 0], second = [0, 0, 0];
    for (let n = 0; n < count; n++) {
      const [a, b] = GP.sample([1, -2], factor, rng);
      sum[0] += a; sum[1] += b; second[0] += a*a; second[1] += a*b; second[2] += b*b;
    }
    const means = sum.map(v => v / count);
    near(means[0], 1, .04); near(means[1], -2, .04);
    near(second[0]/count-means[0]**2, 1, .06); near(second[1]/count-means[0]*means[1], .6, .06); near(second[2]/count-means[1]**2, 2, .08);
  });
  test('訓練入力は重複なし・範囲内で生成', () => {
    const indices = GP.randomIndices(241, 40);
    assert(new Set(indices).size === 40 && indices.every(i => i >= 0 && i < 241), '不正な訓練入力');
  });
})();
if (typeof print === 'function') print(`PASS: ${numericalResults.length} numerical tests`);
