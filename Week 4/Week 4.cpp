#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <limits>
#include <random>
#include <chrono>
#include <algorithm>
#include "RasterSurface.h"
#include "StoneHenge.h" 
#include "StoneHenge_Texture.h"  


const int WIDTH = 500;
const int HEIGHT = 600;
uint32_t raster[HEIGHT][WIDTH];
float depthBufferArray[HEIGHT][WIDTH];

const unsigned int* texture = StoneHenge_pixels;
const unsigned int texture_width = StoneHenge_width;
const unsigned int texture_height = StoneHenge_height;

inline int Convert2Dto1D(int x, int y, int width) {
    return y * width + x;
}

struct Vector3 {
    float x, y, z;

    Vector3(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }


    friend Vector3 operator*(float scalar, const Vector3& vec) {
        return Vector3(vec.x * scalar, vec.y * scalar, vec.z * scalar);
    }

   
    Vector3 operator*(const Vector3& vec) const {
        return Vector3(x * vec.x, y * vec.y, z * vec.z);
    }

    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    float dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3 normalize() const {
        float length = std::sqrt(x * x + y * y + z * z);
        if (length == 0.0f) return Vector3(0, 0, 0);
        return Vector3(x / length, y / length, z / length);
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
};
struct Matrix4x4 {
    float m[4][4];

    Matrix4x4() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    Matrix4x4 operator*(const Matrix4x4& other) const {
        Matrix4x4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; ++k) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }
};

struct Vertex {
    float x, y, z, w;
    float u, v;
    Vector3 position;

    Vertex(float _x, float _y, float _z, float _u = 0.0f, float _v = 0.0f, float _w = 1.0f)
        : x(_x), y(_y), z(_z), u(_u), v(_v), w(_w), position(_x, _y, _z) {}

    Vertex() : x(0.0f), y(0.0f), z(0.0f), u(0.0f), v(0.0f), w(1.0f), position(0, 0, 0) {}

    Vertex operator*(const Matrix4x4& mat) const; // Declaration
};



// Definition of the operator*
Vertex Vertex::operator*(const Matrix4x4& mat) const {
    float newX = x * mat.m[0][0] + y * mat.m[0][1] + z * mat.m[0][2] + w * mat.m[0][3];
    float newY = x * mat.m[1][0] + y * mat.m[1][1] + z * mat.m[1][2] + w * mat.m[1][3];
    float newZ = x * mat.m[2][0] + y * mat.m[2][1] + z * mat.m[2][2] + w * mat.m[2][3];
    float newW = x * mat.m[3][0] + y * mat.m[3][1] + z * mat.m[3][2] + w * mat.m[3][3];
    return Vertex(newX, newY, newZ, u, v, newW);
}


Matrix4x4 translate(float tx, float ty, float tz) {
    Matrix4x4 translation;
    translation.m[0][3] = tx;
    translation.m[1][3] = ty;
    translation.m[2][3] = tz;
    return translation;
}

Matrix4x4 rotateY(float angle) {
    Matrix4x4 rotation;
    float radians = angle * 3.14159265358979323846 / 180.0f;
    rotation.m[0][0] = cos(radians);
    rotation.m[0][2] = sin(radians);
    rotation.m[2][0] = -sin(radians);
    rotation.m[2][2] = cos(radians);
    return rotation;
}

Matrix4x4 rotateX(float angle) {
    Matrix4x4 rotation;
    float radians = angle * 3.14159265358979323846 / 180.0f;
    rotation.m[1][1] = cos(radians);
    rotation.m[1][2] = -sin(radians);
    rotation.m[2][1] = sin(radians);
    rotation.m[2][2] = cos(radians);
    return rotation;
}

Matrix4x4 perspective(float fov, float aspectRatio, float near, float far) {
    Matrix4x4 projection;
    float tanHalfFOV = tan(fov * 3.14159265358979323846 / 180.0f / 2.0f);
    projection.m[0][0] = 1.0f / (aspectRatio * tanHalfFOV);
    projection.m[1][1] = 1.0f / tanHalfFOV;
    projection.m[2][2] = -(far + near) / (far - near);
    projection.m[2][3] = -(2.0f * far * near) / (far - near);
    projection.m[3][2] = -1.0f;
    projection.m[3][3] = 0.0f;
    return projection;
}

Matrix4x4 inverse(const Matrix4x4& mat) {
    Matrix4x4 inv;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            inv.m[i][j] = mat.m[j][i];

    inv.m[0][3] = -(inv.m[0][0] * mat.m[0][3] + inv.m[0][1] * mat.m[1][3] + inv.m[0][2] * mat.m[2][3]);
    inv.m[1][3] = -(inv.m[1][0] * mat.m[0][3] + inv.m[1][1] * mat.m[1][3] + inv.m[1][2] * mat.m[2][3]);
    inv.m[2][3] = -(inv.m[2][0] * mat.m[0][3] + inv.m[2][1] * mat.m[1][3] + inv.m[2][2] * mat.m[2][3]);

    inv.m[3][0] = 0.0f;
    inv.m[3][1] = 0.0f;
    inv.m[3][2] = 0.0f;
    inv.m[3][3] = 1.0f;

    return inv;
}

unsigned int SwapBGRAtoARGB(unsigned int color) {
    unsigned int alpha = (color >> 0) & 0xFF;
    unsigned int blue = (color >> 24) & 0xFF;
    unsigned int green = (color >> 16) & 0xFF;
    unsigned int red = (color >> 8) & 0xFF;

    return (alpha << 24) | (red << 16) | (green << 8) | blue;
}


unsigned int sampleTexture(float u, float v) {
    u = std::max(0.0f, std::min(1.0f, u));
    v = std::max(0.0f, std::min(1.0f, v));

    int texX = static_cast<int>(u * (texture_width - 1));
    int texY = static_cast<int>(v * (texture_height - 1));

    int index = Convert2Dto1D(texX, texY, texture_width);

    unsigned int bgra = texture[index];

    unsigned int argb = SwapBGRAtoARGB(bgra);

    argb = (0xFF << 24) | (argb & 0x00FFFFFF);

    return argb;
}



void clearBuffers() {
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            depthBufferArray[y][x] = std::numeric_limits<float>::infinity();
            raster[y][x] = 0x003366FF;
        }
    }
}

void clearRaster() {
    memset(raster, 0, sizeof(raster));
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++)
            depthBufferArray[y][x] = std::numeric_limits<float>::infinity();
}

void drawPoint(int x, int y, uint32_t color, float depth) {
    if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
        if (depth < depthBufferArray[y][x]) {
            depthBufferArray[y][x] = depth;
            raster[y][x] = color | 0xFF000000;
        }
    }
}

bool barycentric(float px, float py, const Vertex& v0, const Vertex& v1, const Vertex& v2, float& u, float& v, float& w) {
    float denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    if (denom == 0.0f) return false;
    u = ((v1.y - v2.y) * (px - v2.x) + (v2.x - v1.x) * (py - v2.y)) / denom;
    v = ((v2.y - v0.y) * (px - v2.x) + (v0.x - v2.x) * (py - v2.y)) / denom;
    w = 1.0f - u - v;
    return (u >= 0) && (v >= 0) && (w >= 0);
}

bool isBackface(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    float edge1x = v1.x - v0.x;
    float edge1y = v1.y - v0.y;
    float edge1z = v1.z - v0.z;

    float edge2x = v2.x - v0.x;
    float edge2y = v2.y - v0.y;
    float edge2z = v2.z - v0.z;

    float normalX = edge1y * edge2z - edge1z * edge2y;
    float normalY = edge1z * edge2x - edge1x * edge2z;
    float normalZ = edge1x * edge2y - edge1y * edge2x;

    float cameraDirZ = -1.0f;

    return normalZ > 0;
}

void drawFilledTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, uint32_t color) {
    float minX = std::min({ v0.x, v1.x, v2.x });
    float maxX = std::max({ v0.x, v1.x, v2.x });
    float minY = std::min({ v0.y, v1.y, v2.y });
    float maxY = std::max({ v0.y, v1.y, v2.y });

    int xStart = std::max(0, static_cast<int>(std::floor(minX)));
    int xEnd = std::min(WIDTH - 1, static_cast<int>(std::ceil(maxX)));
    int yStart = std::max(0, static_cast<int>(std::floor(minY)));
    int yEnd = std::min(HEIGHT - 1, static_cast<int>(std::ceil(maxY)));

    for (int y = yStart; y <= yEnd; y++) {
        for (int x = xStart; x <= xEnd; x++) {
            float u, v, w;
            if (barycentric(x + 0.5f, y + 0.5f, v0, v1, v2, u, v, w)) {
                float depth = (u * v0.z + v * v1.z + w * v2.z) / (u * v0.w + v * v1.w + w * v2.w);

                if (depth < depthBufferArray[y][x]) {
                    float interpU = u * v0.u + v * v1.u + w * v2.u;
                    float interpV = u * v0.v + v * v1.v + w * v2.v;

                    unsigned int texColor = sampleTexture(interpU, interpV);
                    drawPoint(x, y, texColor, depth);
                }
            }
        }
    }
}


Matrix4x4 buildViewMatrix() {
    Matrix4x4 rotation = rotateX(10.0f);
    Matrix4x4 translation = translate(0.0f, 0.5f, -3.0f);
    Matrix4x4 cameraWorld = rotation * translation;
    Matrix4x4 viewMatrix = inverse(cameraWorld);
    return viewMatrix;
}

Vertex transformVertex(const Vertex& v, const Matrix4x4& world, const Matrix4x4& view, const Matrix4x4& projection) {
    Vertex worldVertex = v * world;
    Vertex viewVertex = worldVertex * view;
    Vertex projectedVertex = viewVertex * projection;
    return projectedVertex;
}

std::vector<Vector3> generateStarField(int numStars) {
    std::vector<Vector3> stars;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (int i = 0; i < numStars; ++i) {
        float x = distribution(generator) * 50.0f;
        float y = distribution(generator) * 50.0f;
        float z = distribution(generator) * 50.0f;
        stars.emplace_back(x, y, z);
    }

    return stars;
}

void drawStars(const std::vector<Vector3>& stars, const Matrix4x4& view, const Matrix4x4& projection) {
    for (const auto& star : stars) {
        Vertex starVertex(star.x, star.y, star.z);
        Vertex projectedStar = transformVertex(starVertex, Matrix4x4(), view, projection);

        if (projectedStar.w != 0.0f) {
            projectedStar.x /= projectedStar.w;
            projectedStar.y /= projectedStar.w;
            projectedStar.z /= projectedStar.w;
        }

        int screenX = static_cast<int>((projectedStar.x * (WIDTH / 2.0f)) + (WIDTH / 2.0f));
        int screenY = static_cast<int>((-projectedStar.y * (HEIGHT / 2.0f)) + (HEIGHT / 2.0f));
        drawPoint(screenX, screenY, 0xFFFFFF, projectedStar.z); 
    }
}

void loadModel(const OBJ_VERT* modelVertices, int numVertices, std::vector<Vertex>& vertexBuffer) {
    for (int i = 0; i < numVertices; ++i) {
        Vertex newVertex(modelVertices[i].pos[0] * 0.1f, modelVertices[i].pos[1] * 0.1f, modelVertices[i].pos[2] * 0.1f, modelVertices[i].uvw[0], modelVertices[i].uvw[1]);
        vertexBuffer.push_back(newVertex);
    }
}


void drawModel(const std::vector<Vertex>& vertexBuffer, const unsigned int* indices, int numIndices, const Matrix4x4& world, const Matrix4x4& view, const Matrix4x4& projection) {
    std::vector<Vertex> transformedVertices;
    for (const auto& v : vertexBuffer) {
        Vertex transformed = transformVertex(v, world, view, projection);
        if (transformed.w != 0.0f) {
            transformed.x /= transformed.w;
            transformed.y /= transformed.w;
            transformed.z /= transformed.w;
        }

        float aspectRatio = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
        transformed.x = transformed.x * (WIDTH / 2.0f) * aspectRatio + (WIDTH / 2.0f);
        transformed.y = transformed.y * (HEIGHT / 2.0f) + (HEIGHT / 2.0f);
        transformedVertices.push_back(transformed);
    }

    for (int i = 0; i < numIndices; i += 3) {
        Vertex v0 = transformedVertices[indices[i]];
        Vertex v1 = transformedVertices[indices[i + 1]];
        Vertex v2 = transformedVertices[indices[i + 2]];

        if (!isBackface(v0, v1, v2)) {
            drawFilledTriangle(v0, v1, v2, 0xFFFFFFFF);
        }
    }
}

int main() {
    if (!RS_Initialize("Star Field and Model Loading", WIDTH, HEIGHT)) {
        std::cerr << "Failed to initialize RasterSurface." << std::endl;
        return -1;
    }

    std::vector<Vector3> stars = generateStarField(3000);
    std::vector<Vertex> vertexBuffer;
    loadModel(StoneHenge_data, 1457, vertexBuffer);


    float angle = 0.0f;
    while (true) {
        clearRaster();
        clearBuffers();

        drawStars(stars, buildViewMatrix(), perspective(90.0f, static_cast<float>(WIDTH) / static_cast<float>(HEIGHT), 0.1f, 20.0f));

        Matrix4x4 translationMatrix = translate(0.0f, 0.0f, 0.0f);
        Matrix4x4 worldMatrix = translationMatrix;

        drawModel(vertexBuffer, StoneHenge_indicies, 2532, worldMatrix, buildViewMatrix(), perspective(90.0f, static_cast<float>(WIDTH) / static_cast<float>(HEIGHT), 0.1f, 20.0f));

        angle += 0.1f;

        if (!RS_Update((unsigned int*)raster, WIDTH * HEIGHT)) {
            break;
        }
    }

    RS_Shutdown();

    return 0;
}