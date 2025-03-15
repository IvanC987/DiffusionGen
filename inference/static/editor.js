let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');

// Original image storage
let originalCtxOriginal = document.createElement('canvas').getContext('2d');
let maskCtxOriginal = document.createElement('canvas').getContext('2d');

let isDrawing = false;
let brushSize = 5;
let eraserMode = false;

// Setup event listeners
document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
document.getElementById('brushSize').addEventListener('change', (e) => brushSize = parseInt(e.target.value));
document.getElementById('eraserMode').addEventListener('change', (e) => eraserMode = e.target.checked);
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function handleImageUpload(e) {
    let file = e.target.files[0];
    let reader = new FileReader();

    reader.onload = function(event) {
        let img = new Image();
        img.onload = function() {
            // Store original image at full resolution
            originalCtxOriginal.canvas.width = img.width;
            originalCtxOriginal.canvas.height = img.height;
            originalCtxOriginal.drawImage(img, 0, 0);

            // Setup mask at original resolution
            maskCtxOriginal.canvas.width = img.width;
            maskCtxOriginal.canvas.height = img.height;

            // Calculate display dimensions
            const container = document.getElementById('canvasContainer');
            const maxWidth = container.clientWidth;
            const maxHeight = container.clientHeight;

            let ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
            ratio = Math.min(ratio, 1); // Prevent upscaling

            const scaledWidth = img.width * ratio;
            const scaledHeight = img.height * ratio;
            const offsetX = (maxWidth - scaledWidth) / 2;
            const offsetY = (maxHeight - scaledHeight) / 2;

            // Set display canvas dimensions
            canvas.width = maxWidth;
            canvas.height = maxHeight;

            // Draw initial scaled image
            ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

            // Store scaling parameters
            window.scaleRatio = ratio;
            window.offsetX = offsetX;
            window.offsetY = offsetY;
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing || !window.scaleRatio) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert to original coordinates
    const originalX = (mouseX - window.offsetX) / window.scaleRatio;
    const originalY = (mouseY - window.offsetY) / window.scaleRatio;
    const brushRadius = brushSize / window.scaleRatio;

    if (eraserMode) {
        // Erase from mask
        maskCtxOriginal.clearRect(
            originalX - brushRadius,
            originalY - brushRadius,
            brushRadius * 2,
            brushRadius * 2
        );
    } else {
        // Draw on mask
        maskCtxOriginal.fillStyle = '#FFFFFF';
        maskCtxOriginal.beginPath();
        maskCtxOriginal.arc(
            originalX,
            originalY,
            brushRadius,
            0,
            Math.PI * 2
        );
        maskCtxOriginal.fill();
    }

    // Redraw display canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(originalCtxOriginal.canvas,
        0, 0, originalCtxOriginal.canvas.width, originalCtxOriginal.canvas.height,
        window.offsetX, window.offsetY,
        originalCtxOriginal.canvas.width * window.scaleRatio,
        originalCtxOriginal.canvas.height * window.scaleRatio
    );

    // Overlay mask
    ctx.globalAlpha = 0.5; // For visual feedback
    ctx.drawImage(maskCtxOriginal.canvas,
        0, 0, maskCtxOriginal.canvas.width, maskCtxOriginal.canvas.height,
        window.offsetX, window.offsetY,
        maskCtxOriginal.canvas.width * window.scaleRatio,
        maskCtxOriginal.canvas.height * window.scaleRatio
    );
    ctx.globalAlpha = 1;
}

// Download button
document.getElementById('downloadBtn').addEventListener('click', () => {
    const binaryCanvas = document.createElement('canvas');
    binaryCanvas.width = originalCtxOriginal.canvas.width;
    binaryCanvas.height = originalCtxOriginal.canvas.height;
    const binaryCtx = binaryCanvas.getContext('2d');

    // Create white background
    binaryCtx.fillStyle = 'white';
    binaryCtx.fillRect(0, 0, binaryCanvas.width, binaryCanvas.height);

    // Get mask data
    const maskData = maskCtxOriginal.getImageData(0, 0,
        binaryCanvas.width, binaryCanvas.height);

    // Draw black where mask exists
    for (let i = 0; i < maskData.data.length; i += 4) {
        if (maskData.data[i + 3] > 0) { // Check alpha channel
            const x = (i / 4) % binaryCanvas.width;
            const y = Math.floor((i / 4) / binaryCanvas.width);
            binaryCtx.fillStyle = 'black';
            binaryCtx.fillRect(x, y, 1, 1);
        }
    }

    document.getElementById('downloadBtn').href = binaryCanvas.toDataURL('image/png');
});