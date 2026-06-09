<template>
  <div class="page">
    <h1>CKKS-MNIST 加密推理演示</h1>
    <p class="subtitle">
      手写或上传一张数字图片，前端处理后由后端调用 CUDA(Phantom-FHE) 做全密文 CKKS 推理。
    </p>

    <div class="cols">
      <!-- 左: 手写画板 -->
      <section class="card">
        <h2>① 手写 / 上传</h2>
        <canvas
          ref="canvas"
          width="280"
          height="280"
          class="board"
          @mousedown="startDraw"
          @mousemove="draw"
          @mouseup="endDraw"
          @mouseleave="endDraw"
        ></canvas>
        <div class="row">
          <button @click="clearBoard">清空</button>
          <label class="upload">
            上传图片
            <input type="file" accept="image/*" @change="onFile" hidden />
          </label>
        </div>
        <label class="invert">
          <input type="checkbox" v-model="invert" /> 反相 (白底黑字时勾选)
        </label>
      </section>

      <!-- 右: 结果 -->
      <section class="card">
        <h2>② 推理结果</h2>
        <button class="primary" :disabled="loading" @click="predict">
          {{ loading ? "密文推理中…(约 10s)" : "开始加密推理" }}
        </button>

        <div v-if="error" class="error">{{ error }}</div>

        <div v-if="result" class="result">
          <div class="pred">
            预测数字：<b>{{ result.prediction }}</b>
            <span class="conf">（置信度 {{ (probs[result.prediction] * 100).toFixed(1) }}%）</span>
          </div>
          <div class="meta" v-if="result.slot_count">
            slot_count={{ result.slot_count }}，mid levels={{ result.mid_levels }}
          </div>
          <div class="bars">
            <div v-for="(p, i) in probs" :key="i" class="bar-row">
              <span class="bar-label">{{ i }}</span>
              <div class="bar-track">
                <div
                  class="bar-fill"
                  :class="{ top: i === result.prediction }"
                  :style="{ width: (p * 100) + '%' }"
                ></div>
              </div>
              <span class="bar-val">{{ (p * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from "vue";

const canvas = ref(null);
let ctx = null;
let drawing = false;

const invert = ref(false);
const loading = ref(false);
const error = ref("");
const result = ref(null);

// 把后端返回的 10 个 logit 做 softmax, 得到各类别概率。
// 减去最大值是数值稳定写法, 避免 exp 溢出。
const probs = computed(() => {
  if (!result.value) return [];
  const logits = result.value.logits;
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
});

onMounted(() => {
  ctx = canvas.value.getContext("2d");
  clearBoard();
});

function clearBoard() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 280, 280);
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#fff";
  result.value = null;
  error.value = "";
}

function pos(e) {
  const r = canvas.value.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}
function startDraw(e) {
  drawing = true;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}
function draw(e) {
  if (!drawing) return;
  const p = pos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}
function endDraw() {
  drawing = false;
}

function onFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 280, 280);
    ctx.drawImage(img, 0, 0, 280, 280);
  };
  img.src = URL.createObjectURL(file);
  invert.value = true; // 上传图通常白底黑字
}

async function predict() {
  loading.value = true;
  error.value = "";
  result.value = null;
  try {
    const dataUrl = canvas.value.toDataURL("image/png");
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: dataUrl, invert: invert.value }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "请求失败");
    result.value = data;
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}
</script>

<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, "Segoe UI", sans-serif; background: #0f172a; color: #e2e8f0; }
.page { max-width: 760px; margin: 0 auto; padding: 32px 16px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.subtitle { color: #94a3b8; margin: 0 0 24px; font-size: 14px; }
.cols { display: flex; gap: 16px; flex-wrap: wrap; }
.card { flex: 1; min-width: 300px; background: #1e293b; border-radius: 12px; padding: 16px; }
.card h2 { font-size: 16px; margin: 0 0 12px; }
.board { background: #000; border-radius: 8px; cursor: crosshair; touch-action: none; }
.row { display: flex; gap: 8px; margin-top: 12px; }
button { background: #334155; color: #e2e8f0; border: none; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 14px; }
button:hover { background: #475569; }
button.primary { background: #2563eb; width: 100%; padding: 12px; font-weight: 600; }
button.primary:disabled { background: #1e3a8a; cursor: not-allowed; }
.upload { background: #334155; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-size: 14px; }
.invert { display: block; margin-top: 12px; font-size: 13px; color: #94a3b8; }
.error { margin-top: 12px; color: #f87171; font-size: 14px; word-break: break-all; }
.result { margin-top: 16px; }
.pred { font-size: 18px; margin-bottom: 4px; }
.conf { font-size: 13px; color: #94a3b8; }
.meta { color: #94a3b8; font-size: 12px; margin-bottom: 12px; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 12px; }
.bar-label { width: 14px; text-align: right; color: #94a3b8; }
.bar-track { flex: 1; background: #0f172a; border-radius: 4px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; background: #475569; transition: width 0.3s; }
.bar-fill.top { background: #22c55e; }
.bar-val { width: 52px; text-align: right; font-variant-numeric: tabular-nums; }
</style>
