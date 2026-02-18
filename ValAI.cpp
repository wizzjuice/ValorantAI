#include <thread>
#include <cmath>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <iomanip>
#include <random>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")

constexpr int CIRCLE_RADIUS = 80;
constexpr int CIRCLE_THICKNESS = 1;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT rc;
        GetClientRect(hwnd, &rc);
        int centerX = (rc.right - rc.left) / 2;
        int centerY = (rc.bottom - rc.top) / 2;

        HPEN pen = CreatePen(PS_SOLID, CIRCLE_THICKNESS, RGB(255, 255, 255));
        HGDIOBJ oldPen = SelectObject(hdc, pen);
        HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));

        Ellipse(hdc, centerX - CIRCLE_RADIUS, centerY - CIRCLE_RADIUS,
            centerX + CIRCLE_RADIUS, centerY + CIRCLE_RADIUS);

        SelectObject(hdc, oldBrush);
        SelectObject(hdc, oldPen);
        DeleteObject(pen);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

class ArduinoHID {
private:
    HANDLE hSerial;

public:
    bool connect(const char* portName) {
        hSerial = CreateFileA(portName,
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, NULL);

        if (hSerial == INVALID_HANDLE_VALUE) {
            std::cerr << "Failed to connect to " << portName << std::endl;
            return false;
        }

        DCB dcbSerialParams = { 0 };
        dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

        if (!GetCommState(hSerial, &dcbSerialParams)) {
            CloseHandle(hSerial);
            return false;
        }

        dcbSerialParams.BaudRate = CBR_115200;
        dcbSerialParams.ByteSize = 8;
        dcbSerialParams.StopBits = ONESTOPBIT;
        dcbSerialParams.Parity = NOPARITY;

        if (!SetCommState(hSerial, &dcbSerialParams)) {
            CloseHandle(hSerial);
            return false;
        }

        COMMTIMEOUTS timeouts = { 0 };
        timeouts.ReadIntervalTimeout = 50;
        timeouts.ReadTotalTimeoutConstant = 50;
        timeouts.ReadTotalTimeoutMultiplier = 10;
        timeouts.WriteTotalTimeoutConstant = 50;
        timeouts.WriteTotalTimeoutMultiplier = 10;

        if (!SetCommTimeouts(hSerial, &timeouts)) {
            CloseHandle(hSerial);
            return false;
        }

        std::cout << "Connected to Arduino on " << portName << std::endl;
        Sleep(2000);
        return true;
    }

    void moveMouseRelative(int dx, int dy) {
        dx = std::max(-127, std::min(127, dx));
        dy = std::max(-127, std::min(127, dy));
        unsigned char cmd[3] = { 'M', (unsigned char)dx, (unsigned char)dy };
        DWORD bytesWritten;
        WriteFile(hSerial, cmd, 3, &bytesWritten, NULL);
    }

    void click(int button = 1) {
        unsigned char cmd[3] = { 'C', (unsigned char)button, 0 };
        DWORD bytesWritten;
        WriteFile(hSerial, cmd, 3, &bytesWritten, NULL);
    }

    void disconnect() {
        if (hSerial != INVALID_HANDLE_VALUE) {
            CloseHandle(hSerial);
        }
    }

    ~ArduinoHID() {
        disconnect();
    }
};

struct FrameData {
    std::vector<uint8_t> pixels;
    UINT width;
    UINT height;
    UINT pitch;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct TRTDestroy {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

nvinfer1::ICudaEngine* loadEngine(const std::string& engineFile, Logger& logger) {
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open engine file: " << engineFile << std::endl;
        return nullptr;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return nullptr;
    }

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return nullptr;
    }

    return engine;
}

void preprocessFrame(const FrameData& frame, std::vector<float>& output, int targetWidth, int targetHeight) {
    float scaleX = static_cast<float>(frame.width) / targetWidth;
    float scaleY = static_cast<float>(frame.height) / targetHeight;

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < targetHeight; ++y) {
            for (int x = 0; x < targetWidth; ++x) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;

                int x1 = static_cast<int>(srcX);
                int y1 = static_cast<int>(srcY);
                int x2 = std::min(x1 + 1, (int)frame.width - 1);
                int y2 = std::min(y1 + 1, (int)frame.height - 1);

                float fx = srcX - x1;
                float fy = srcY - y1;

                auto getPixel = [&](int px, int py, int channel) -> float {
                    int idx = py * frame.pitch + px * 4;
                    uint8_t pixel;
                    if (channel == 0) pixel = frame.pixels[idx + 2];
                    else if (channel == 1) pixel = frame.pixels[idx + 1];
                    else pixel = frame.pixels[idx + 0];
                    return pixel / 255.0f;
                    };

                float p11 = getPixel(x1, y1, c);
                float p21 = getPixel(x2, y1, c);
                float p12 = getPixel(x1, y2, c);
                float p22 = getPixel(x2, y2, c);

                float value = (1 - fx) * (1 - fy) * p11 +
                    fx * (1 - fy) * p21 +
                    (1 - fx) * fy * p12 +
                    fx * fy * p22;

                int dstIndex = c * targetHeight * targetWidth + y * targetWidth + x;
                output[dstIndex] = value;
            }
        }
    }
}

struct Detection {
    float x, y, w, h;
    float confidence;
    int classId;
};

struct CoordinateConverter {
    float inferenceWidth;
    float inferenceHeight;
    float screenWidth;
    float screenHeight;
    float scaleX;
    float scaleY;

    CoordinateConverter(float infW, float infH, float scrW, float scrH)
        : inferenceWidth(infW), inferenceHeight(infH),
        screenWidth(scrW), screenHeight(scrH) {
        scaleX = screenWidth / inferenceWidth;
        scaleY = screenHeight / inferenceHeight;
    }

    void inferenceToScreen(float infX, float infY, float& scrX, float& scrY) const {
        scrX = infX * scaleX;
        scrY = infY * scaleY;
    }

    void screenToInference(float scrX, float scrY, float& infX, float& infY) const {
        infX = scrX / scaleX;
        infY = scrY / scaleY;
    }
};

struct PersistentTargetTracker {
    float smoothedErrorX = 0.0f;
    float smoothedErrorY = 0.0f;

    float velocityX = 0.0f;
    float velocityY = 0.0f;
    float lastScreenX = 0.0f;
    float lastScreenY = 0.0f;

    float smoothedVelX = 0.0f;
    float smoothedVelY = 0.0f;

    bool hasLockedTarget = false;
    Detection lockedDetection;
    float targetConfidence = 0.0f;
    int framesTracked = 0;
    int framesMissed = 0;

    const float CONFIDENCE_GAIN = 0.25f;
    const float CONFIDENCE_DECAY = 0.10f;
    const float MIN_LOCK_CONFIDENCE = 0.55f;
    const float SWITCH_THRESHOLD = 3.0f;
    const int MAX_FRAMES_MISSING = 15;
    const float POSITION_MATCH_THRESHOLD = 100.0f;

    void reset() {
        smoothedErrorX = 0.0f;
        smoothedErrorY = 0.0f;
        velocityX = 0.0f;
        velocityY = 0.0f;
        smoothedVelX = 0.0f;
        smoothedVelY = 0.0f;
        hasLockedTarget = false;
        targetConfidence = 0.0f;
        framesTracked = 0;
        framesMissed = 0;
    }

    void updateVelocity(float currentScreenX, float currentScreenY) {
        if (framesTracked > 0) {
            velocityX = currentScreenX - lastScreenX;
            velocityY = currentScreenY - lastScreenY;
        }
        lastScreenX = currentScreenX;
        lastScreenY = currentScreenY;
    }

    bool isSameTarget(const Detection& det) const {
        if (!hasLockedTarget) return false;
        float dx = det.x - lockedDetection.x;
        float dy = det.y - lockedDetection.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        return dist < POSITION_MATCH_THRESHOLD && det.classId == lockedDetection.classId;
    }

    float calculateTargetScore(const Detection& det, float distanceFromCenter, float maxDistance) const {
        float sizeScore = (det.w * det.h) / 10000.0f;
        float distNorm = std::min(distanceFromCenter / maxDistance, 1.0f);
        float proximityScore = 1.0f - distNorm;

        return sizeScore * 0.4f + proximityScore * 0.6f;
    }

    void updateWithDetection(const Detection& det, float distanceFromCenter, float maxDistance) {
        float currentScore = calculateTargetScore(det, distanceFromCenter, maxDistance);

        if (!hasLockedTarget) {
            lockedDetection = det;
            hasLockedTarget = true;
            targetConfidence = 0.5f;
            framesTracked = 1;
            framesMissed = 0;
        }
        else if (isSameTarget(det)) {
            lockedDetection = det;
            targetConfidence = std::min(1.0f, targetConfidence + CONFIDENCE_GAIN);
            framesTracked++;
            framesMissed = 0;
        }
        else {
            float centerX = 320.0f;
            float centerY = 320.0f;
            float lockedDx = lockedDetection.x - centerX;
            float lockedDy = lockedDetection.y - centerY;
            float lockedDist = std::sqrt(lockedDx * lockedDx + lockedDy * lockedDy);
            float lockedScore = calculateTargetScore(lockedDetection, lockedDist, maxDistance);

            if (currentScore > lockedScore * SWITCH_THRESHOLD ||
                targetConfidence < MIN_LOCK_CONFIDENCE) {
                lockedDetection = det;
                hasLockedTarget = true;
                targetConfidence = 0.5f;
                framesTracked = 1;
                framesMissed = 0;
                velocityX = 0.0f;
                velocityY = 0.0f;
                smoothedVelX = 0.0f;
                smoothedVelY = 0.0f;
            }
        }
    }

    void updateWithoutDetection() {
        if (!hasLockedTarget) return;
        targetConfidence -= CONFIDENCE_DECAY;
        framesMissed++;
        if (targetConfidence < 0.0f || framesMissed > MAX_FRAMES_MISSING) {
            reset();
        }
    }

    bool shouldAim() const {
        return hasLockedTarget && targetConfidence >= MIN_LOCK_CONFIDENCE &&
            framesTracked >= 2;
    }

    Detection getCurrentTarget() const {
        return lockedDetection;
    }
};

struct VirtualHitbox {
    float x, y, w, h;
    float aimX, aimY;
};

VirtualHitbox createVirtualHitbox(const Detection& det, float headExtensionFactor = 0.45f) {
    VirtualHitbox vhb;
    float headExtension = det.h * headExtensionFactor;

    vhb.w = det.w;
    vhb.h = det.h + headExtension;
    vhb.x = det.x;
    vhb.y = det.y - (headExtension / 2.0f);

    float aimHeadOffset;
    if (det.h > 60.0f) {
        aimHeadOffset = 0.4f;
    }
    else if (det.h > 40.0f) {
        aimHeadOffset = 0.35f;
    }
    else {
        aimHeadOffset = 0.25f;
    }

    vhb.aimX = det.x;
    vhb.aimY = det.y - (det.h * aimHeadOffset);

    return vhb;
}

struct AdaptiveAimConfig {
    float pixelsPerCount = 2.2f;
    float deadzone = 0.8f;
    float maxSpeed = 127.0f;

    float minGain = 0.35f;
    float maxGain = 0.70f;
    float gainTransitionDistance = 40.0f;

    bool addHumanization = true;
    float jitterAmount = 0.2f;

    float accelerationBoost = 1.05f;
    bool useAccelerationBias = true;
    float accelerationMinDistance = 100.0f;

    float omega = 0.55f;
    float zeta = 1.0f;

    float maxPredictionFrames = 1.5f;
    float minSpeedForPrediction = 3.0f;

    float calculateGain(float errorMagnitude) const {
        if (errorMagnitude < gainTransitionDistance) {
            return minGain;
        }
        float t = std::min(1.0f,
            (errorMagnitude - gainTransitionDistance) / gainTransitionDistance);
        return minGain + (maxGain - minGain) * t;
    }
};

void addHumanization(int& dx, int& dy, float amount) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-amount, amount);
    dx += static_cast<int>(dist(rng));
    dy += static_cast<int>(dist(rng));
}

ArduinoHID arduino;

void aimAtTargetAdaptive(float targetScreenX, float targetScreenY,
    float screenWidth, float screenHeight,
    PersistentTargetTracker& tracker, const AdaptiveAimConfig& config) {

    float centerX = screenWidth / 2.0f;
    float centerY = screenHeight / 2.0f;

    float errorX = targetScreenX - centerX;
    float errorY = targetScreenY - centerY;
    float errorMagnitude = std::sqrt(errorX * errorX + errorY * errorY);

    tracker.updateVelocity(targetScreenX, targetScreenY);

    float speed = std::sqrt(tracker.velocityX * tracker.velocityX +
        tracker.velocityY * tracker.velocityY);

    if (speed > config.minSpeedForPrediction) {
        float predictionFrames = std::min(speed / 20.0f, config.maxPredictionFrames);
        errorX += tracker.velocityX * predictionFrames;
        errorY += tracker.velocityY * predictionFrames;
    }

    float dt = 1.0f;
    float omega = config.omega;
    float zeta = config.zeta;

    float accelX = omega * omega * (errorX - tracker.smoothedErrorX) -
        2.0f * zeta * omega * tracker.smoothedVelX;
    float accelY = omega * omega * (errorY - tracker.smoothedErrorY) -
        2.0f * zeta * omega * tracker.smoothedVelY;

    tracker.smoothedVelX += accelX * dt;
    tracker.smoothedVelY += accelY * dt;
    tracker.smoothedErrorX += tracker.smoothedVelX * dt;
    tracker.smoothedErrorY += tracker.smoothedVelY * dt;

    float moveX = tracker.smoothedErrorX / config.pixelsPerCount;
    float moveY = tracker.smoothedErrorY / config.pixelsPerCount;

    float gain = config.calculateGain(errorMagnitude);
    moveX *= gain;
    moveY *= gain;

    if (config.useAccelerationBias &&
        tracker.framesTracked <= 3 &&
        errorMagnitude > config.accelerationMinDistance) {
        float boostFactor = config.accelerationBoost *
            (1.0f - (tracker.framesTracked / 3.0f));
        moveX *= boostFactor;
        moveY *= boostFactor;
    }

    float moveMagnitude = std::sqrt(moveX * moveX + moveY * moveY);
    if (moveMagnitude > config.maxSpeed) {
        float scale = config.maxSpeed / moveMagnitude;
        moveX *= scale;
        moveY *= scale;
    }

    float errorInCounts = errorMagnitude / config.pixelsPerCount;
    if (errorInCounts < config.deadzone) {
        return;
    }

    int dx = static_cast<int>(std::round(moveX));
    int dy = static_cast<int>(std::round(moveY));

    if (config.addHumanization && errorMagnitude > 10.0f) {
        float jitterScale = (errorMagnitude < 30.0f) ? 0.3f : 1.0f;
        addHumanization(dx, dy, config.jitterAmount * jitterScale);
    }

    dx = std::max(-127, std::min(127, dx));
    dy = std::max(-127, std::min(127, dy));

    if (dx != 0 || dy != 0) {
        arduino.moveMouseRelative(dx, dy);
    }
}

struct DetectionFilter {
    struct TrackedDetection {
        Detection det;
        int framesSeen;
        int framesLost;
    };

    std::vector<TrackedDetection> tracked;
    int minFramesRequired = 2;
    int maxFramesLost = 5;

    std::vector<Detection> filter(const std::vector<Detection>& newDetections) {
        for (auto& track : tracked) {
            track.framesLost++;
        }

        for (const auto& det : newDetections) {
            bool matched = false;

            for (auto& track : tracked) {
                float dx = det.x - track.det.x;
                float dy = det.y - track.det.y;
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist < 70.0f && det.classId == track.det.classId) {
                    track.det = det;
                    track.framesSeen++;
                    track.framesLost = 0;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                tracked.push_back({ det, 1, 0 });
            }
        }

        tracked.erase(
            std::remove_if(tracked.begin(), tracked.end(),
                [this](const TrackedDetection& t) {
                    return t.framesLost > maxFramesLost;
                }),
            tracked.end()
        );

        std::vector<Detection> confirmed;
        for (const auto& track : tracked) {
            if (track.framesSeen >= minFramesRequired) {
                confirmed.push_back(track.det);
            }
        }

        return confirmed;
    }
};

std::vector<Detection> nonMaximumSuppression(std::vector<Detection>& detections, float iouThreshold) {
    if (detections.empty()) return {};

    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        float x1_i = detections[i].x - detections[i].w / 2;
        float y1_i = detections[i].y - detections[i].h / 2;
        float x2_i = detections[i].x + detections[i].w / 2;
        float y2_i = detections[i].y + detections[i].h / 2;
        float area_i = detections[i].w * detections[i].h;

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].classId != detections[j].classId) continue;

            float x1_j = detections[j].x - detections[j].w / 2;
            float y1_j = detections[j].y - detections[j].h / 2;
            float x2_j = detections[j].x + detections[j].w / 2;
            float y2_j = detections[j].y + detections[j].h / 2;

            float xx1 = std::max(x1_i, x1_j);
            float yy1 = std::max(y1_i, y1_j);
            float xx2 = std::min(x2_i, x2_j);
            float yy2 = std::min(y2_i, y2_j);

            float w_inter = std::max(0.0f, xx2 - xx1);
            float h_inter = std::max(0.0f, yy2 - yy1);
            float area_inter = w_inter * h_inter;

            float area_j = detections[j].w * detections[j].h;
            float iou = area_inter / (area_i + area_j - area_inter);

            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

bool virtualHitboxIntersectsCircle(const VirtualHitbox& vhb,
    float inferenceWidth, float inferenceHeight,
    float circleRadiusInference) {

    float centerX = inferenceWidth / 2.0f;
    float centerY = inferenceHeight / 2.0f;

    float boxLeft = vhb.x - vhb.w / 2.0f;
    float boxRight = vhb.x + vhb.w / 2.0f;
    float boxTop = vhb.y - vhb.h / 2.0f;
    float boxBottom = vhb.y + vhb.h / 2.0f;

    float closestX = std::max(boxLeft, std::min(centerX, boxRight));
    float closestY = std::max(boxTop, std::min(centerY, boxBottom));

    float dx = closestX - centerX;
    float dy = closestY - centerY;
    float distanceSquared = dx * dx + dy * dy;

    return distanceSquared <= (circleRadiusInference * circleRadiusInference);
}

struct TargetInfo {
    Detection* detection;
    VirtualHitbox virtualBox;
    bool isValid;
};

TargetInfo findBestTargetInCircle(std::vector<Detection>& detections,
    const CoordinateConverter& converter,
    int targetClassId, float circleRadiusPixels) {

    TargetInfo bestTarget;
    bestTarget.detection = nullptr;
    bestTarget.isValid = false;

    float circleRadiusInference = circleRadiusPixels / converter.scaleX;

    float centerX = converter.inferenceWidth / 2.0f;
    float centerY = converter.inferenceHeight / 2.0f;
    float maxDistance = std::sqrt(centerX * centerX + centerY * centerY);
    float maxScore = 0.0f;

    for (auto& det : detections) {
        if (det.classId != targetClassId) continue;

        VirtualHitbox vhb = createVirtualHitbox(det);

        if (!virtualHitboxIntersectsCircle(vhb, converter.inferenceWidth,
            converter.inferenceHeight,
            circleRadiusInference)) {
            continue;
        }

        float dx = det.x - centerX;
        float dy = det.y - centerY;
        float distFromCenter = std::sqrt(dx * dx + dy * dy);

        float sizeScore = (det.w * det.h) / 10000.0f;
        float distNorm = std::min(distFromCenter / maxDistance, 1.0f);
        float proximityScore = 1.0f - distNorm;
        float totalScore = sizeScore * 0.4f + proximityScore * 0.6f;

        if (totalScore > maxScore) {
            maxScore = totalScore;
            bestTarget.detection = &det;
            bestTarget.virtualBox = vhb;
            bestTarget.isValid = true;
        }
    }

    return bestTarget;
}

void overlayThread(std::atomic<bool>& keepRunning)
{
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    WNDCLASS wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"OverlayWindowClass";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
        wc.lpszClassName, L"", WS_POPUP,
        0, 0, screenWidth, screenHeight,
        nullptr, nullptr, hInstance, nullptr
    );

    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    MSG msg;
    while (keepRunning)
    {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    DestroyWindow(hwnd);
    UnregisterClass(wc.lpszClassName, hInstance);
}

int main() {
    HRESULT hr;

    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    D3D_FEATURE_LEVEL featureLevel;

    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        0, nullptr, 0, D3D11_SDK_VERSION, &d3dDevice,
        &featureLevel, &d3dContext);
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device! HRESULT: 0x" << std::hex << hr << std::endl;
        return -1;
    }

    std::cout << "D3D11 device created.\n";

    Logger logger;
    nvinfer1::ICudaEngine* engine = loadEngine("C:\\ONNX\\best.engine", logger);
    if (!engine) {
        std::cerr << "Failed to load engine!\n";
        d3dContext->Release();
        d3dDevice->Release();
        return -1;
    }

    TRTUniquePtr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context!\n";
        delete engine;
        d3dContext->Release();
        d3dDevice->Release();
        return -1;
    }

    int32_t nbIOTensors = engine->getNbIOTensors();
    std::string inputName, outputName;
    nvinfer1::Dims inputDims, outputDims;

    for (int32_t i = 0; i < nbIOTensors; ++i) {
        const char* tensorName = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(tensorName);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName = tensorName;
            inputDims = engine->getTensorShape(tensorName);
            std::cout << "Input tensor: " << tensorName << std::endl;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName = tensorName;
            outputDims = engine->getTensorShape(tensorName);
            std::cout << "Output tensor: " << tensorName << std::endl;
        }
    }

    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) {
        inputSize *= inputDims.d[i];
    }
    inputSize *= sizeof(float);

    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) {
        outputSize *= outputDims.d[i];
    }
    outputSize *= sizeof(float);

    std::cout << "Input size: " << inputSize / sizeof(float) << " floats\n";
    std::cout << "Output size: " << outputSize / sizeof(float) << " floats\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_input, inputSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc input failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaStreamDestroy(stream);
        return -1;
    }

    cudaStatus = cudaMalloc((void**)&d_output, outputSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc output failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_input);
        cudaStreamDestroy(stream);
        return -1;
    }

    context->setTensorAddress(inputName.c_str(), d_input);
    context->setTensorAddress(outputName.c_str(), d_output);

    if (!arduino.connect("\\\\.\\COM")) {
        std::cerr << "Failed to connect to Arduino. Check Device Manager for COM port.\n";
        return -1;
    }

    IDXGIDevice* dxgiDevice = nullptr;
    hr = d3dDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    if (FAILED(hr)) {
        std::cerr << "Failed to get DXGI device\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        d3dContext->Release();
        d3dDevice->Release();
        return -1;
    }

    IDXGIAdapter* dxgiAdapter = nullptr;
    dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);

    IDXGIOutput* dxgiOutput = nullptr;
    hr = dxgiAdapter->EnumOutputs(0, &dxgiOutput);
    if (FAILED(hr)) {
        std::cerr << "Failed to enumerate outputs\n";
        dxgiAdapter->Release();
        dxgiDevice->Release();
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        d3dContext->Release();
        d3dDevice->Release();
        return -1;
    }

    IDXGIOutput1* dxgiOutput1 = nullptr;
    dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);

    IDXGIOutputDuplication* deskDupl = nullptr;
    hr = dxgiOutput1->DuplicateOutput(d3dDevice, &deskDupl);
    if (FAILED(hr)) {
        std::cerr << "Failed to duplicate output! HRESULT: 0x" << std::hex << hr << std::endl;
        dxgiOutput1->Release();
        dxgiOutput->Release();
        dxgiAdapter->Release();
        dxgiDevice->Release();
        cudaFree(d_input);
        cudaFree(d_output);
        cudaStreamDestroy(stream);
        d3dContext->Release();
        d3dDevice->Release();
        return -1;
    }

    std::queue<FrameData> frameQueue;
    std::mutex queueMutex;
    std::condition_variable frameAvailable;
    std::atomic<bool> keepRunning(true);
    std::atomic<bool> needsReinit(false);

    std::thread captureThread([&]() {
        IDXGIOutputDuplication* localDeskDupl = deskDupl;

        while (keepRunning) {
            if (needsReinit) {
                if (localDeskDupl) {
                    localDeskDupl->Release();
                    localDeskDupl = nullptr;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                hr = dxgiOutput1->DuplicateOutput(d3dDevice, &localDeskDupl);
                if (SUCCEEDED(hr)) {
                    std::cout << "Desktop duplication recovered successfully!\n";
                    needsReinit = false;
                }
                else {
                    std::cerr << "Failed to recover, retrying...\n";
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
            }

            IDXGIResource* desktopResource = nullptr;
            DXGI_OUTDUPL_FRAME_INFO frameInfo = {};
            HRESULT hr = localDeskDupl->AcquireNextFrame(16, &frameInfo, &desktopResource);
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) continue;
            if (FAILED(hr)) {
                if (hr == DXGI_ERROR_ACCESS_LOST) {
                    std::cerr << "Desktop duplication access lost, reinitializing...\n";
                    needsReinit = true;
                    continue;
                }
                continue;
            }

            ID3D11Texture2D* acquiredDesktopImage = nullptr;
            desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredDesktopImage);

            D3D11_TEXTURE2D_DESC desc;
            acquiredDesktopImage->GetDesc(&desc);

            D3D11_TEXTURE2D_DESC descCopy = desc;
            descCopy.Usage = D3D11_USAGE_STAGING;
            descCopy.BindFlags = 0;
            descCopy.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            descCopy.MiscFlags = 0;

            ID3D11Texture2D* cpuTexture = nullptr;
            d3dDevice->CreateTexture2D(&descCopy, nullptr, &cpuTexture);

            d3dContext->CopyResource(cpuTexture, acquiredDesktopImage);

            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(d3dContext->Map(cpuTexture, 0, D3D11_MAP_READ, 0, &mapped))) {
                FrameData frame;
                frame.width = desc.Width;
                frame.height = desc.Height;
                frame.pitch = mapped.RowPitch;
                frame.pixels.resize(frame.pitch * frame.height);
                memcpy(frame.pixels.data(), mapped.pData, frame.pitch * frame.height);

                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    while (frameQueue.size() >= 1) frameQueue.pop();
                    frameQueue.push(std::move(frame));
                }
                frameAvailable.notify_one();
                d3dContext->Unmap(cpuTexture, 0);
            }

            cpuTexture->Release();
            acquiredDesktopImage->Release();
            desktopResource->Release();
            localDeskDupl->ReleaseFrame();
        }

        if (localDeskDupl && localDeskDupl != deskDupl) {
            localDeskDupl->Release();
        }
        });

    std::thread processingThread([&]() {
        std::vector<float> inputFloat(inputSize / sizeof(float));
        std::vector<float> output(outputSize / sizeof(float));

        PersistentTargetTracker tracker;
        DetectionFilter detectionFilter;

        AdaptiveAimConfig config;
        config.pixelsPerCount = 2.2f;
        config.minGain = 0.35f;
        config.maxGain = 0.70f;
        config.accelerationBoost = 1.05f;
        config.deadzone = 0.8f;
        config.omega = 0.55f;

        while (keepRunning) {
            std::unique_lock<std::mutex> lock(queueMutex);
            frameAvailable.wait(lock, [&]() { return !frameQueue.empty() || !keepRunning; });
            if (!keepRunning) break;

            FrameData frame = std::move(frameQueue.front());
            frameQueue.pop();
            lock.unlock();

            CoordinateConverter converter(640.0f, 640.0f,
                static_cast<float>(frame.width),
                static_cast<float>(frame.height));

            preprocessFrame(frame, inputFloat, 640, 640);

            cudaMemcpyAsync(d_input, inputFloat.data(), inputSize, cudaMemcpyHostToDevice, stream);

            bool success = context->enqueueV3(stream);
            if (!success) {
                std::cerr << "Inference failed\n";
                continue;
            }

            cudaMemcpyAsync(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int numOutputElements = output.size();
            int numAttributes = 8;
            int numPredictions = numOutputElements / numAttributes;

            float confThreshold = 0.40f;
            float nmsThreshold = 0.5f;
            std::vector<Detection> detections;

            for (int i = 0; i < numPredictions; ++i) {
                float cx = output[0 * numPredictions + i];
                float cy = output[1 * numPredictions + i];
                float w = output[2 * numPredictions + i];
                float h = output[3 * numPredictions + i];

                float maxConf = 0.0f;
                int maxClass = -1;
                int numClasses = numAttributes - 4;

                for (int c = 0; c < numClasses; ++c) {
                    float conf = output[(4 + c) * numPredictions + i];
                    if (conf > maxConf) {
                        maxConf = conf;
                        maxClass = c;
                    }
                }

                if (maxConf > confThreshold) {
                    Detection det;
                    det.x = cx;
                    det.y = cy;
                    det.w = w;
                    det.h = h;
                    det.confidence = maxConf;
                    det.classId = maxClass;
                    detections.push_back(det);
                }
            }

            std::vector<Detection> nmsDetections = nonMaximumSuppression(detections, nmsThreshold);
            /*
            nmsDetections.erase(
                std::remove_if(nmsDetections.begin(), nmsDetections.end(),
                    [](const Detection& d) {
                        return d.w < 15.0f || d.h < 15.0f;
                    }),
                nmsDetections.end()
            );
            */
            std::vector<Detection> confirmedDetections = detectionFilter.filter(nmsDetections);

            TargetInfo targetInfo = findBestTargetInCircle(
                confirmedDetections,
                converter,
                1,
                static_cast<float>(CIRCLE_RADIUS)
            );

            if (targetInfo.isValid) {
                float centerX = 320.0f;
                float centerY = 320.0f;
                float dx = targetInfo.detection->x - centerX;
                float dy = targetInfo.detection->y - centerY;
                float distFromCenter = std::sqrt(dx * dx + dy * dy);
                float maxDistance = std::sqrt(centerX * centerX + centerY * centerY);

                tracker.updateWithDetection(*targetInfo.detection, distFromCenter, maxDistance);

                float screenX, screenY;
                converter.inferenceToScreen(targetInfo.virtualBox.aimX,
                    targetInfo.virtualBox.aimY,
                    screenX, screenY);

                if (tracker.shouldAim()) {
                    aimAtTargetAdaptive(screenX, screenY,
                        static_cast<float>(frame.width),
                        static_cast<float>(frame.height),
                        tracker, config);
                }
            }
            else {
                tracker.updateWithoutDetection();
            }
        }
        });

    std::thread overlay(overlayThread, std::ref(keepRunning));

    std::cout << "Press BACKSPACE to exit...\n";

    while (!(GetAsyncKeyState(VK_BACK) & 0x8000)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    keepRunning = false;
    frameAvailable.notify_all();

    captureThread.join();
    processingThread.join();
    overlay.join();

    context.reset();
    cudaStreamDestroy(stream);
    deskDupl->Release();
    dxgiOutput1->Release();
    dxgiOutput->Release();
    dxgiAdapter->Release();
    dxgiDevice->Release();
    d3dContext->Release();
    d3dDevice->Release();
    cudaFree(d_input);
    cudaFree(d_output);
    delete engine;

    std::cout << "Cleanup complete.\n";
    return 0;
};
