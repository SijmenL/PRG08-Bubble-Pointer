import {FilesetResolver, HandLandmarker} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

document.addEventListener('DOMContentLoaded', init);

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

let canvasWidth;
let canvasHeight;

let lastVideoTime = -1;
let results = undefined;
let pose;
let sortedResults;

let lastCircleSpawnTime = performance.now();
const circleSpawnInterval = 1000;

let networkPredictions = document.getElementById('predictions')

let video = document.getElementById("webcam");
let canvasElement = document.getElementById("output_canvas");
let canvasCtx = canvasElement.getContext("2d");

const bubbles = [];

const nn = ml5.neuralNetwork({task: 'classification', debug: true})
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
}

function init() {
    createHandLandmarker();

    const hasGetUserMedia = () => {
        let _a;
        return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
    };

    if (hasGetUserMedia()) {
        enableWebcamButton = document.getElementById("webcamButton");
        enableWebcamButton.addEventListener("click", enableCam);
    } else {
        console.warn("getUserMedia() is not supported by your browser");
    }

    nn.load(modelDetails, () => console.log("het model is geladen!"))

    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);
}

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1
    });
};


function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }
    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "Start Scannen";
    } else {

        if (bubbles.length < 1) {
            for (let i = 0; i < 25; i++) {
                createCircle();
            }
        }
        webcamRunning = true;
        enableWebcamButton.innerText = "Stop Scannen";
    }

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}


function setCanvasSize() {
    canvasElement.style.width = window.innerWidth + 'px';
    canvasElement.style.height = window.innerHeight + 'px';
    canvasElement.width = window.innerWidth;
    canvasElement.height = window.innerHeight;

    // Update canvas width and height variables
    canvasWidth = canvasElement.width;
    canvasHeight = canvasElement.height;
}

async function predictWebcam() {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({runningMode: "VIDEO"});
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.save();
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            pose = landmarks.flatMap(coord => [coord.x, coord.y, coord.z]);

            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#01ffff",
                lineWidth: 1
            });

            drawLandmarks(canvasCtx, landmarks, {color: "#00c5ff", lineWidth: 1});

            if (pose !== undefined) {
                canvasCtx.beginPath();
                canvasCtx.arc(pose[24] * canvasWidth, pose[25] * canvasHeight, 5, 0, Math.PI * 2);
                canvasCtx.fillStyle = 'blue';
                canvasCtx.fill();
                canvasCtx.closePath();
            }

            const currentTime = performance.now();
            if (currentTime - lastCircleSpawnTime >= circleSpawnInterval) {
                createCircle();
                lastCircleSpawnTime = currentTime;
            }

            bubbles.forEach((circle, index) => {
                drawCircle(circle);

                // Update circle position
                circle.x += circle.dx;
                circle.y += circle.dy;

                // Check for collision with canvasElement edges
                if (circle.x + circle.radius > canvasElement.width || circle.x - circle.radius < 0) {
                    circle.dx = -circle.dx; // Reverse horizontal velocity
                }
                if (circle.y + circle.radius > canvasElement.height || circle.y - circle.radius < 0) {
                    circle.dy = -circle.dy; // Reverse vertical velocity
                }

                const distanceToPose = Math.sqrt((circle.x - pose[24] * canvasWidth) ** 2 + (circle.y - pose[25] * canvasHeight) ** 2);
                if (distanceToPose <= circle.radius && sortedResults[0].label === 'pointing') {
                    if (circle.color !== '#2596be') {
                        console.log("player died");
                        stopGame();
                    } else {
                        bubbles.splice(index, 1);
                    }
                }
            });

            let predictions = await nn.classify(pose);
            sortedResults = predictions.sort((a, b) => b.confidence - a.confidence);

            let fragment = document.createDocumentFragment();

            sortedResults.forEach(prediction => {
                let p = document.createElement('p');
                p.innerText = `${prediction.label}: ${prediction.confidence}`;
                fragment.appendChild(p);
            });

            networkPredictions.innerHTML = '';
            networkPredictions.appendChild(fragment);
        }
    }

    // Draw circles
    bubbles.forEach(circle => {
        drawCircle(circle);

        // Update circle position
        circle.x += circle.dx;
        circle.y += circle.dy;

        // Check for collision with canvasElement edges
        if (circle.x + circle.radius > canvasElement.width || circle.x - circle.radius < 0) {
            circle.dx = -circle.dx; // Reverse horizontal velocity
        }
        if (circle.y + circle.radius > canvasElement.height || circle.y - circle.radius < 0) {
            circle.dy = -circle.dy; // Reverse vertical velocity
        }
    });

    canvasCtx.restore();


    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function createCircle() {
    // Create new circle
    const radius = Math.random() * 20 + 10;
    const x = Math.random() * (canvasElement.width - 2 * radius) + radius;
    const y = Math.random() * (canvasElement.height - 2 * radius) + radius;
    const dx = (Math.random() - 0.5) * 2;
    const dy = (Math.random() - 0.5) * 2;

    let circleColor;
    if (Math.random() < 0.2) {
        circleColor = '#be2525';
    } else {
        circleColor = '#2596be';
    }
    const color = circleColor

    bubbles.push({ x, y, radius, dx, dy, color });
}

function drawCircle(circle) {
    canvasCtx.beginPath();
    canvasCtx.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);
    canvasCtx.fillStyle = 'transparent';
    canvasCtx.strokeStyle = circle.color;
    canvasCtx.lineWidth = 3;
    canvasCtx.stroke();
    canvasCtx.fill();
    canvasCtx.closePath();
}

function stopGame() {
    webcamRunning = false;
}