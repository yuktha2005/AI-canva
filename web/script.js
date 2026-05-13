const videoElement = document.getElementById('input_video');
const overlayCanvas = document.getElementById('overlay_canvas');
const overlayCtx = overlayCanvas.getContext('2d');
const drawingCanvas = document.getElementById('drawing_canvas');
const drawingCtx = drawingCanvas.getContext('2d');
const uiCanvas = document.getElementById('ui_canvas');
const uiCtx = uiCanvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const loadingScreen = document.getElementById('loading');

// Set actual processing resolution
const WIDTH = 1280;
const HEIGHT = 720;
overlayCanvas.width = WIDTH; overlayCanvas.height = HEIGHT;
drawingCanvas.width = WIDTH; drawingCanvas.height = HEIGHT;
uiCanvas.width = WIDTH; uiCanvas.height = HEIGHT;

const colors = [
    { name: 'Eraser', hex: '#000000', isEraser: true },
    { name: 'Blue', hex: '#3b82f6', isEraser: false },
    { name: 'Yellow', hex: '#eab308', isEraser: false },
    { name: 'Purple', hex: '#a855f7', isEraser: false },
    { name: 'Green', hex: '#22c55e', isEraser: false }
];

let currentColorIdx = 1; // Default to Blue
let prevX = 0;
let prevY = 0;

// Helper: draw a rounded rectangle path (works in ALL browsers)
function drawRoundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

// Draw the top palette UI
function drawUI() {
    uiCtx.clearRect(0, 0, WIDTH, HEIGHT);
    
    // Draw Top Toolbar background
    uiCtx.fillStyle = "rgba(15, 23, 42, 0.6)";
    uiCtx.fillRect(0, 0, WIDTH, 100);
    
    const boxWidth = WIDTH / colors.length;
    
    for (let i = 0; i < colors.length; i++) {
        const c = colors[i];
        const x1 = i * boxWidth;
        const y1 = 15;
        const width = boxWidth - 20;
        const height = 70;
        const centerX = x1 + 10 + width / 2;
        const centerY = y1 + height / 2;
        
        // Draw Button Box
        uiCtx.fillStyle = c.isEraser ? "#cbd5e1" : c.hex;
        drawRoundedRect(uiCtx, x1 + 10, y1, width, height, 12);
        uiCtx.fill();
        
        // Highlight outline if selected
        if (i === currentColorIdx) {
            uiCtx.strokeStyle = "#ffffff";
            uiCtx.lineWidth = 4;
            uiCtx.stroke();
            
            // Draw a tiny indicator circle
            uiCtx.beginPath();
            uiCtx.arc(centerX, y1 - 4, 6, 0, 2 * Math.PI);
            uiCtx.fillStyle = "#ffffff";
            uiCtx.fill();
        }
        
        // Text
        uiCtx.fillStyle = c.isEraser ? "#0f172a" : "#ffffff";
        uiCtx.font = "bold 26px Inter, sans-serif";
        uiCtx.textAlign = "center";
        uiCtx.textBaseline = "middle";
        if (!c.isEraser) {
            uiCtx.shadowColor = "rgba(0,0,0,0.3)";
            uiCtx.shadowBlur = 4;
        } else {
            uiCtx.shadowColor = "transparent";
        }
        uiCtx.fillText(c.name, centerX, centerY);
        uiCtx.shadowColor = "transparent";
    }
}

drawUI(); // Initial draw

// Clear Canvas
clearBtn.addEventListener('click', () => {
    drawingCtx.clearRect(0, 0, WIDTH, HEIGHT);
});

// Setup MediaPipe Hands
const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 1280,
    height: 720
});

// Start camera and hide loading screen
camera.start().then(() => {
    loadingScreen.style.opacity = '0';
    setTimeout(() => loadingScreen.style.display = 'none', 300);
}).catch(err => {
    loadingScreen.innerHTML = `<p style="color: #ef4444;">Error accessing camera: ${err.message}</p>`;
});

function onResults(results) {
    // Clear the overlay canvas each frame (transparent so video shows through)
    overlayCtx.save();
    overlayCtx.clearRect(0, 0, WIDTH, HEIGHT);
    
    // Mirror the overlay context to match the mirrored video
    overlayCtx.translate(WIDTH, 0);
    overlayCtx.scale(-1, 1);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            // Draw hand skeleton on the overlay
            drawConnectors(overlayCtx, landmarks, HAND_CONNECTIONS, {color: 'rgba(255,255,255,0.6)', lineWidth: 2});
            drawLandmarks(overlayCtx, landmarks, {color: '#94a3b8', lineWidth: 1, radius: 3});
            
            const indexTip = landmarks[8];
            const indexPip = landmarks[6];
            const middleTip = landmarks[12];
            const middlePip = landmarks[10];
            const thumbTip = landmarks[4];
            const thumbIp = landmarks[3];
            
            // Check finger states
            const indexUp = indexTip.y < indexPip.y;
            const middleUp = middleTip.y < middlePip.y;
            const thumbUp = thumbTip.y < thumbIp.y;
            
            // Map X coordinates (mirrored for drawing canvas) and Y coordinates
            const x = WIDTH - (indexTip.x * WIDTH);
            const y = indexTip.y * HEIGHT;
            
            const thumbX = WIDTH - (thumbTip.x * WIDTH);
            const thumbY = thumbTip.y * HEIGHT;
            
            // SELECT MODE: Index & Middle up OR Thumb up near the top
            if ((indexUp && middleUp) || (thumbUp && thumbY < 120)) {
                prevX = 0; prevY = 0;
                
                let selX = (indexUp && middleUp) ? x : thumbX;
                let selY = (indexUp && middleUp) ? y : thumbY;
                
                // Draw cursor on overlay (in the mirrored context, use raw landmark coords)
                let renderX = (indexUp && middleUp) ? indexTip.x * WIDTH : thumbTip.x * WIDTH;
                let renderY = (indexUp && middleUp) ? indexTip.y * HEIGHT : thumbTip.y * HEIGHT;
                
                overlayCtx.beginPath();
                overlayCtx.arc(renderX, renderY, 15, 0, 2 * Math.PI);
                overlayCtx.fillStyle = colors[currentColorIdx].isEraser ? "#ffffff" : colors[currentColorIdx].hex;
                overlayCtx.fill();
                overlayCtx.lineWidth = 2;
                overlayCtx.strokeStyle = "#000";
                overlayCtx.stroke();
                
                // If cursor is in toolbar area, select color
                if (selY < 100) {
                    const boxWidth = WIDTH / colors.length;
                    const idx = Math.floor(selX / boxWidth);
                    if (idx >= 0 && idx < colors.length && idx !== currentColorIdx) {
                        currentColorIdx = idx;
                        drawUI();
                    }
                }
            } 
            // DRAW MODE: Index up, Middle down
            else if (indexUp && !middleUp) {
                // Draw cursor on overlay
                overlayCtx.beginPath();
                overlayCtx.arc(indexTip.x * WIDTH, indexTip.y * HEIGHT, 10, 0, 2 * Math.PI);
                overlayCtx.fillStyle = colors[currentColorIdx].isEraser ? "#ffffff" : colors[currentColorIdx].hex;
                overlayCtx.fill();
                
                if (prevX === 0 && prevY === 0) {
                    prevX = x;
                    prevY = y;
                }
                
                // Draw on persistent drawing canvas
                drawingCtx.beginPath();
                drawingCtx.moveTo(prevX, prevY);
                drawingCtx.lineTo(x, y);
                
                if (colors[currentColorIdx].isEraser) {
                    drawingCtx.globalCompositeOperation = 'destination-out';
                    drawingCtx.lineWidth = 50;
                    drawingCtx.strokeStyle = "rgba(0,0,0,1)";
                } else {
                    drawingCtx.globalCompositeOperation = 'source-over';
                    drawingCtx.lineWidth = 8;
                    drawingCtx.strokeStyle = colors[currentColorIdx].hex;
                }
                
                drawingCtx.lineCap = 'round';
                drawingCtx.lineJoin = 'round';
                drawingCtx.stroke();
                
                prevX = x;
                prevY = y;
            } else {
                prevX = 0; prevY = 0;
            }
        }
    } else {
        prevX = 0; prevY = 0;
    }
    
    overlayCtx.restore();
}
