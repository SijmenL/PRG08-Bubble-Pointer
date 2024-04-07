import {FilesetResolver, HandLandmarker} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

document.addEventListener('DOMContentLoaded', init);

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let enableWebcamButtonEndscreen;
let webcamRunning = false;

let endScreen = document.getElementById('endScreen');
let probabilityDisplay = document.getElementById('probabilityDisplay')

let canvasWidth;
let canvasHeight;

let lastVideoTime = -1;
let results = undefined;
let pose;
let sortedResults;
let gameStarted = false;

let info = document.getElementById('info')
let pointsDisplay = document.getElementById('points')
let timeDisplay = document.getElementById('timer')

let points = 0;
let totalTime = 45
let time = totalTime;

let lastCircleSpawnTime = performance.now();
const circleSpawnInterval = 1000;

let networkPredictions = document.getElementById('predictions')

let menu = document.getElementById('startMenu')

let video = document.getElementById("webcam");
let canvasElement = document.getElementById("output_canvas");
let canvasCtx = canvasElement.getContext("2d");

const bubbles = [];

const nn = ml5.neuralNetwork({task: 'classification', debug: true})
const modelDetails = {
    model: '../model/model.json',
    metadata: '../model/model_meta.json',
    weights: '../model/model.weights.bin'
}

function init() {
    createHandLandmarker();

    timeDisplay.innerText = time;

    const hasGetUserMedia = () => {
        let _a;
        return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia);
    };

    if (hasGetUserMedia()) {
        enableWebcamButton = document.getElementById("webcamButton");
        enableWebcamButton.addEventListener("click", enableCam);

        enableWebcamButtonEndscreen = document.getElementById("webcamButtonEndscreen");
        enableWebcamButtonEndscreen.addEventListener("click", function () {
            location.reload();
        });
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

        if (bubbles.length < 1) {
            for (let i = 0; i < 25; i++) {
                createCircle();
            }
        }
        webcamRunning = true;
        enableWebcamButton.innerText = "Loading...";

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            if (!gameStarted) {
                startGame();
                gameStarted = true;
            }
            predictWebcam();
        });
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

function startGame() {
    menu.classList.add('fade-out');
    canvasElement.classList.add('fade-in')
    info.classList.add('fade-in')

    let timerCountdown = setInterval(() => {
        time--;
        timeDisplay.innerText = time;

        if (time < 10) {
            timeDisplay.style.color = 'red';
        }

        if (time <= 0) {
            clearInterval(timerCountdown);
            stopGame();
        }
    }, 1000);
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
                color: "rgba(255,255,255,0.25)",
                lineWidth: 1
            });

            drawLandmarks(canvasCtx, landmarks, {color: "rgba(255,255,255,0.5)", lineWidth: 1});

            if (pose !== undefined) {
                canvasCtx.beginPath();
                canvasCtx.arc(pose[24] * canvasWidth, pose[25] * canvasHeight, 7, 0, Math.PI * 2);
                canvasCtx.fillStyle = '#2596be';
                canvasCtx.fill();
                canvasCtx.closePath();

                canvasCtx.beginPath();
                canvasCtx.arc(pose[60] * canvasWidth, pose[61] * canvasHeight, 7, 0, Math.PI * 2);
                canvasCtx.fillStyle = '#be2525';
                canvasCtx.fill();
                canvasCtx.closePath();

                canvasCtx.beginPath();
                canvasCtx.arc(pose[12] * canvasWidth, pose[13] * canvasHeight, 7, 0, Math.PI * 2);
                canvasCtx.fillStyle = '#25be2d';
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

                if (sortedResults) {
                    if (sortedResults[0].label === 'index') {
                        let distanceToPose = Math.sqrt((circle.x - pose[24] * canvasWidth) ** 2 + (circle.y - pose[25] * canvasHeight) ** 2);
                        if (distanceToPose <= circle.radius) {
                            if (circle.color === '#2596be') {
                                points = points + Math.ceil(circle.radius * 2);
                                bubbles.splice(index, 1);
                                pointsDisplay.innerText = points;

                                if (points > 0) {
                                    pointsDisplay.style.color = 'white';
                                }
                            } else {
                                points = points - 75;
                                bubbles.splice(index, 1);
                                pointsDisplay.innerText = points;

                                if (points < 0) {
                                    pointsDisplay.style.color = 'red';
                                }
                            }
                        }
                    }

                    if (sortedResults[0].label === 'pinky') {
                        let distanceToPose = Math.sqrt((circle.x - pose[60] * canvasWidth) ** 2 + (circle.y - pose[61] * canvasHeight) ** 2);
                        if (distanceToPose <= circle.radius) {
                            if (circle.color === '#be2525') {
                                points = points + Math.ceil(circle.radius * 2);
                                bubbles.splice(index, 1);
                                pointsDisplay.innerText = points;

                                if (points > 0) {
                                    pointsDisplay.style.color = 'white';
                                }
                            } else {
                                points = points - 75;
                                bubbles.splice(index, 1);
                                pointsDisplay.innerText = points;

                                if (points < 0) {
                                    pointsDisplay.style.color = 'red';
                                }
                            }
                        }
                    }

                    if (sortedResults[0].label === 'thumb') {
                        let distanceToPose = Math.sqrt((circle.x - pose[12] * canvasWidth) ** 2 + (circle.y - pose[13] * canvasHeight) ** 2);
                        if (distanceToPose <= circle.radius) {
                            if (circle.color === '#25be2d') {
                                points = points + Math.ceil(circle.radius * 2);
                                bubbles.splice(index, 1);
                                pointsDisplay.innerText = points;

                                if (points > 0) {
                                    pointsDisplay.style.color = 'white';
                                }
                            } else {
                                points = points - 75;
                                bubbles.splice(index, 1);
                                pointsDisplay.innerText = points;

                                if (points < 0) {
                                    pointsDisplay.style.color = 'red';
                                }
                            }
                        }

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

            if (sortedResults[0].label === 'index') {
                probabilityDisplay.innerText = '☝️';
            }
            if (sortedResults[0].label === 'pinky') {
                probabilityDisplay.innerHTML = ''
                let img = document.createElement('img')
                img.src = 'img/pinky.png';
                img.classList.add('emoji-img')
                probabilityDisplay.appendChild(img)
            }
            if (sortedResults[0].label === 'thumb') {
                probabilityDisplay.innerText = '👍';
            }
            if (sortedResults[0].label === 'nothing') {
                probabilityDisplay.innerText = '🚫';
            }

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
    let randomColor = Math.random()
    if (randomColor < 0.333) {
        circleColor = '#be2525';
    }
    if (randomColor > 0.333 && randomColor < 0.666) {
        circleColor = '#2596be';
    }
    if (randomColor > 0.666) {
        circleColor = '#25be2d';
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
    canvasElement.classList.remove('fade-in')
    canvasElement.classList.add('fade-out')
    info.classList.remove('fade-in')
    info.classList.add('fade-out')
    endScreen.classList.remove('no-view')
    endScreen.classList.add('fade-in')
    webcamRunning = false;
    gameStarted = false;

    document.getElementById('scoreEndscreen').innerText = points
}