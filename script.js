// ============================================================================
// Dos realidades — Ejercicio 02 (DPPI 2026)
//
// Condición estricta de activación por 2 personas:
// Utiliza MediaPipe FaceDetector ultraligero (BlazeFace Short Range, solo 220 KB)
// diseñado específicamente para detectar múltiples rostros simultáneamente frente
// a la cámara web sin tiempos de espera ni fallos de descarga.
//
// - 0 o 1 persona: Ambos sistemas permanecen completamente en negro y sin ningún
//   texto escrito en pantalla.
//
// - 2 personas:
//     SISTEMA A: Máscaras biométricas de color (Cian para Persona 1, Magenta para Persona 2).
//     SISTEMA B: Histograma óptico en tiempo real (RGB + Luminancia).
// ============================================================================

// ---------------------------------------------------------------------------
// Configuración general
// ---------------------------------------------------------------------------

const VISION_BUNDLE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Modelo ultraligero BlazeFace (~220 KB), descarga en menos de 1 segundo
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

// Colores asignados a cada persona en el Sistema A
const PERSON_COLORS = [
  { r: 0, g: 245, b: 212, hex: "#00f5d4", name: "Cian" },     // Persona 1
  { r: 247, g: 37, b: 133, hex: "#f72585", name: "Magenta" }, // Persona 2
];

// Resolución para el muestreo del histograma (160x120 = 19,200 píxeles, rápido y fluido)
const HIST_SAMPLE_W = 160;
const HIST_SAMPLE_H = 120;

// ---------------------------------------------------------------------------
// Referencias DOM
// ---------------------------------------------------------------------------

const video = document.getElementById("video");
const canvasA = document.getElementById("canvasA");
const ctxA = canvasA.getContext("2d");
const canvasB = document.getElementById("canvasB");
const ctxB = canvasB.getContext("2d");
const hiddenSample = document.getElementById("hiddenSample");
const ctxHidden = hiddenSample.getContext("2d", { willReadFrequently: true });

const startBtn = document.getElementById("startBtn");
const cameraBtnWrapper = document.getElementById("cameraBtnWrapper");
const statusMsg = document.getElementById("statusMsg");
const statA = document.getElementById("statA");
const statB = document.getElementById("statB");

// Configurar tamaño del canvas oculto de muestreo
hiddenSample.width = HIST_SAMPLE_W;
hiddenSample.height = HIST_SAMPLE_H;

// ---------------------------------------------------------------------------
// Estado interno
// ---------------------------------------------------------------------------

let faceDetector = null;
let running = false;
let lastVideoTime = -1;

// ---------------------------------------------------------------------------
// Arranque
// ---------------------------------------------------------------------------

startBtn.addEventListener("click", start);

async function start() {
  startBtn.disabled = true;

  setStatus("Solicitando acceso a la cámara…");
  try {
    await initCamera();
  } catch (err) {
    console.error(err);
    setStatus(
      "No se pudo acceder a la cámara: " +
        (err && err.message ? err.message : "revisa los permisos de cámara y vuelve a intentarlo.")
    );
    startBtn.disabled = false;
    return;
  }

  // Dejar ambos canvas en negro puro inmediatamente
  clearToBlack(ctxA, canvasA.width, canvasA.height);
  clearToBlack(ctxB, canvasB.width, canvasB.height);

  running = true;
  cameraBtnWrapper.classList.add("is-live");
  requestAnimationFrame(renderLoop);

  setStatus("Cargando modelo de visión artificial…");
  try {
    await initFaceDetector();
    setStatus("Cámara y modelo activos.");
    // Desvanecer el mensaje tras 2 segundos para máxima limpieza visual
    setTimeout(() => {
      setStatus("");
    }, 2000);
  } catch (err) {
    console.error("Error al cargar FaceDetector:", err);
    setStatus("Error al cargar: " + (err && err.message ? err.message : String(err)));
  }
}

function setStatus(text) {
  if (statusMsg) statusMsg.textContent = text;
}

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  await new Promise((resolve) => {
    if (video.readyState >= 2) return resolve();
    video.onloadedmetadata = () => resolve();
  });

  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvasA.width = w;
  canvasA.height = h;
  canvasB.width = w;
  canvasB.height = h;
}

async function initFaceDetector() {
  const { FaceDetector, FilesetResolver } = await import(VISION_BUNDLE_URL);
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

  const options = {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.35,
  };

  try {
    faceDetector = await FaceDetector.createFromOptions(vision, options);
  } catch (err) {
    console.warn("Fallo con GPU, reintentando con CPU…", err);
    options.baseOptions.delegate = "CPU";
    faceDetector = await FaceDetector.createFromOptions(vision, options);
  }
}

// ---------------------------------------------------------------------------
// Loop principal
// ---------------------------------------------------------------------------

function renderLoop(timestampMs) {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    let result = null;
    if (faceDetector) {
      try {
        result = faceDetector.detectForVideo(video, timestampMs);
      } catch (err) {
        console.error("Error en inferencia de video:", err);
      }
    }

    const detections = result && result.detections ? result.detections : [];
    
    // Condición estricta: solo se activa si hay al menos dos personas/rostros
    const hasTwoPeople = detections.length >= 2;

    drawSystemA(hasTwoPeople, detections);
    drawSystemB(hasTwoPeople);
  }

  requestAnimationFrame(renderLoop);
}

// ---------------------------------------------------------------------------
// Utilidad: Limpiar a negro absoluto sin texto
// ---------------------------------------------------------------------------

function clearToBlack(ctx, w, h) {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, w, h);
}

// ---------------------------------------------------------------------------
// SISTEMA A — Máscaras Biométricas por Color (2 Personas)
// ---------------------------------------------------------------------------

function drawSystemA(hasTwoPeople, detections) {
  const w = canvasA.width;
  const h = canvasA.height;

  // Si no hay dos personas: negro absoluto y sin ningún texto escrito
  if (!hasTwoPeople || detections.length < 2) {
    clearToBlack(ctxA, w, h);
    if (statA) statA.textContent = "";
    return;
  }

  // Fondo negro puro
  clearToBlack(ctxA, w, h);
  if (statA) statA.textContent = "Individualidad activa · Dos presencias (Cian & Magenta)";

  // Renderizar la máscara de cada persona
  for (let i = 0; i < 2; i++) {
    const detection = detections[i];
    const color = PERSON_COLORS[i];
    drawFaceMask(ctxA, detection, color, w, h);
  }
}

function drawFaceMask(ctx, detection, color, w, h) {
  ctx.save();
  ctx.globalCompositeOperation = "screen";

  const b = detection.boundingBox;
  if (!b) {
    ctx.restore();
    return;
  }

  const bx = b.originX !== undefined ? b.originX : (b.xmin !== undefined ? b.xmin : 0);
  const by = b.originY !== undefined ? b.originY : (b.ymin !== undefined ? b.ymin : 0);
  const bw = b.width;
  const bh = b.height;

  const isNorm = bw <= 1.0;
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;

  const px = isNorm ? bx * w : (bx / vw) * w;
  const py = isNorm ? by * h : (by / vh) * h;
  const pw = isNorm ? bw * w : (bw / vw) * w;
  const ph = isNorm ? bh * h : (bh / vh) * h;

  // Centro y radio circular del rostro
  const cx = px + pw / 2;
  const cy = py + ph / 2;
  const radius = Math.max(pw, ph) * 0.65;

  // 1. Silueta / Máscara circular rellena con gradiente orgánico
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(15, radius), 0, Math.PI * 2);

  const grad = ctx.createRadialGradient(cx, cy, radius * 0.1, cx, cy, radius);
  grad.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, 0.30)`);
  grad.addColorStop(0.7, `rgba(${color.r}, ${color.g}, ${color.b}, 0.62)`);
  grad.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0.95)`);
  ctx.fillStyle = grad;
  ctx.fill();

  // 2. Contorno exterior de la máscara con resplandor neón
  ctx.lineWidth = 3;
  ctx.strokeStyle = color.hex;
  ctx.shadowColor = color.hex;
  ctx.shadowBlur = 24;
  ctx.stroke();

  // 3. Rasgos biométricos / puntos clave (ojos, nariz, boca)
  if (detection.keypoints && detection.keypoints.length >= 4) {
    ctx.shadowBlur = 14;
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = color.hex;
    ctx.lineWidth = 2;

    const kps = detection.keypoints.map((kp) => ({
      x: kp.x <= 1.0 ? kp.x * w : (kp.x / vw) * w,
      y: kp.y <= 1.0 ? kp.y * h : (kp.y / vh) * h,
    }));

    // Ojo derecho y Ojo izquierdo (índices 0 y 1)
    if (kps[0] && kps[1]) {
      // Línea de la mirada
      ctx.beginPath();
      ctx.moveTo(kps[0].x, kps[0].y);
      ctx.lineTo(kps[1].x, kps[1].y);
      ctx.stroke();

      // Círculos de ojos
      [kps[0], kps[1]].forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }

    // Nariz (índice 2)
    if (kps[2]) {
      ctx.beginPath();
      ctx.arc(kps[2].x, kps[2].y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Boca (índice 3)
    if (kps[3]) {
      ctx.beginPath();
      ctx.arc(kps[3].x, kps[3].y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  ctx.restore();
}

// ---------------------------------------------------------------------------
// SISTEMA B — Histograma de Imagen en Tiempo Real (2 Personas)
// ---------------------------------------------------------------------------

function drawSystemB(hasTwoPeople) {
  const w = canvasB.width;
  const h = canvasB.height;

  // Si no hay dos personas: negro absoluto y sin ningún texto escrito
  if (!hasTwoPeople) {
    clearToBlack(ctxB, w, h);
    if (statB) statB.textContent = "";
    return;
  }

  if (statB) statB.textContent = "Colectividad activa · Espectro dinámico";

  // Muestrear fotograma de la cámara en el canvas oculto
  ctxHidden.drawImage(video, 0, 0, HIST_SAMPLE_W, HIST_SAMPLE_H);
  const frame = ctxHidden.getImageData(0, 0, HIST_SAMPLE_W, HIST_SAMPLE_H).data;
  const totalPixels = HIST_SAMPLE_W * HIST_SAMPLE_H;

  // Distribución de frecuencias para 256 niveles (0..255)
  const histR = new Float32Array(256);
  const histG = new Float32Array(256);
  const histB = new Float32Array(256);
  const histLuma = new Float32Array(256);

  let sumLuma = 0;
  for (let i = 0; i < frame.length; i += 4) {
    const r = frame[i];
    const g = frame[i + 1];
    const b = frame[i + 2];
    const luma = Math.round(0.299 * r + 0.587 * g + 0.114 * b);

    histR[r]++;
    histG[g]++;
    histB[b]++;
    histLuma[luma]++;
    sumLuma += luma;
  }

  const meanLuma = Math.round(sumLuma / totalPixels);

  // Encontrar el pico más alto para escalar proporcionalmente
  let peak = 1;
  for (let i = 0; i < 256; i++) {
    if (histLuma[i] > peak) peak = histLuma[i];
    if (histR[i] > peak) peak = histR[i];
    if (histG[i] > peak) peak = histG[i];
    if (histB[i] > peak) peak = histB[i];
  }

  // --- Renderizado del Histograma ---
  clearToBlack(ctxB, w, h);

  const padX = 40;
  const padTop = 50;
  const padBottom = 46;
  const chartW = w - padX * 2;
  const chartH = h - padTop - padBottom;
  const baseY = padTop + chartH;

  // Cuadrícula técnica de fondo
  ctxB.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctxB.lineWidth = 1;

  // Líneas horizontales de referencia (25%, 50%, 75%, 100%)
  for (let step = 1; step <= 4; step++) {
    const y = padTop + chartH * (1 - step / 4);
    ctxB.beginPath();
    ctxB.moveTo(padX, y);
    ctxB.lineTo(padX + chartW, y);
    ctxB.stroke();
  }

  // Líneas verticales de referencia (0, 64, 128, 192, 255)
  const xTicks = [0, 64, 128, 192, 255];
  ctxB.font = "11px monospace";
  ctxB.fillStyle = "rgba(255, 255, 255, 0.4)";
  ctxB.textAlign = "center";

  xTicks.forEach((val) => {
    const x = padX + (val / 255) * chartW;
    ctxB.beginPath();
    ctxB.moveTo(x, padTop);
    ctxB.lineTo(x, baseY + 6);
    ctxB.stroke();
    ctxB.fillText(val.toString(), x, baseY + 20);
  });

  // Base del gráfico
  ctxB.strokeStyle = "rgba(255, 255, 255, 0.25)";
  ctxB.beginPath();
  ctxB.moveTo(padX, baseY);
  ctxB.lineTo(padX + chartW, baseY);
  ctxB.stroke();

  // Función para dibujar una curva de área del canal
  function drawChannel(histArray, strokeColor, fillColor) {
    ctxB.save();
    ctxB.beginPath();
    ctxB.moveTo(padX, baseY);

    for (let i = 0; i < 256; i++) {
      const x = padX + (i / 255) * chartW;
      const normalizedHeight = (histArray[i] / peak) * chartH;
      const y = baseY - normalizedHeight;
      ctxB.lineTo(x, y);
    }

    ctxB.lineTo(padX + chartW, baseY);
    ctxB.closePath();

    ctxB.fillStyle = fillColor;
    ctxB.fill();

    ctxB.strokeStyle = strokeColor;
    ctxB.lineWidth = 1.6;
    ctxB.stroke();
    ctxB.restore();
  }

  // Dibujar canales con mezcla aditiva 'screen'
  ctxB.save();
  ctxB.globalCompositeOperation = "screen";

  // Canal Rojo (R)
  drawChannel(histR, "#ff4365", "rgba(255, 67, 101, 0.28)");
  // Canal Verde (G)
  drawChannel(histG, "#00f59b", "rgba(0, 245, 155, 0.28)");
  // Canal Azul (B)
  drawChannel(histB, "#38bdf8", "rgba(56, 189, 248, 0.28)");

  // Canal Luminancia (Luma) por encima en blanco brillante
  ctxB.globalCompositeOperation = "source-over";
  ctxB.beginPath();
  for (let i = 0; i < 256; i++) {
    const x = padX + (i / 255) * chartW;
    const y = baseY - (histLuma[i] / peak) * chartH;
    if (i === 0) ctxB.moveTo(x, y);
    else ctxB.lineTo(x, y);
  }
  ctxB.strokeStyle = "rgba(255, 255, 255, 0.9)";
  ctxB.lineWidth = 2;
  ctxB.shadowColor = "#ffffff";
  ctxB.shadowBlur = 8;
  ctxB.stroke();
  ctxB.restore();

  // Encabezado técnico del visor en la parte superior del canvas
  ctxB.font = "600 12px 'Segoe UI', monospace";
  ctxB.fillStyle = "#f1ecf7";
  ctxB.textAlign = "left";
  ctxB.fillText("COLECTIVIDAD · ESPECTRO COMPARTIDO", padX, 26);

  ctxB.textAlign = "right";
  ctxB.fillStyle = "rgba(255, 255, 255, 0.7)";
  ctxB.fillText(`EXPOSICIÓN: ${meanLuma}/255 (${Math.round((meanLuma / 255) * 100)}%)`, padX + chartW, 26);
}
