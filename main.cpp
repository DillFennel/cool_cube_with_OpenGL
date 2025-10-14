#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <vector>

// Шейдеры
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D ourTexture;
uniform int faceType; // 0 = обычная грань, 1 = грань с котом
uniform float time;   // Добавим время для анимации

void main()
{
    if (faceType == 1) {
        // Грань с котом - используем текстуру
        vec4 texColor = texture(ourTexture, TexCoord);
        
        // Добавляем мерцание глаз с помощью времени
        vec2 eye1 = vec2(0.4, 0.55);
        vec2 eye2 = vec2(0.6, 0.55);
        float dist1 = distance(TexCoord, eye1);
        float dist2 = distance(TexCoord, eye2);
        
        if (dist1 < 0.03 || dist2 < 0.03) {
            // Мерцающий эффект для глаз
            float blink = sin(time * 5.0) * 0.5 + 0.5;
            texColor.rgb = mix(texColor.rgb, vec3(1.0, 1.0, 1.0), blink * 0.3);
        }
        
        FragColor = texColor;
    } else {
        // Обычные грани - яркие веселые цвета
        vec3 color;
        if (TexCoord.x < 0.5) {
            if (TexCoord.y < 0.5) color = vec3(1.0, 0.5, 0.0); // оранжевый
            else color = vec3(0.0, 1.0, 0.5); // бирюзовый
        } else {
            if (TexCoord.y < 0.5) color = vec3(1.0, 0.0, 1.0); // пурпурный
            else color = vec3(1.0, 1.0, 0.0); // желтый
        }
        FragColor = vec4(color, 1.0);
    }
}
)";

// Функция компиляции шейдера
unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Ошибка компиляции шейдера: " << infoLog << std::endl;
    }
    return shader;
}

// Функция создания шейдерной программы
unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Ошибка линковки шейдерной программы: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

// Создание текстуры с веселым котом
unsigned int createHappyCatTexture() {
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    // Параметры текстуры
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Создаем изображение веселого кота (128x128 пикселей для лучшего качества)
    const int width = 128;
    const int height = 128;
    std::vector<unsigned char> image(width * height * 4); // RGBA
    
    // Цвета
    unsigned char bgColor[3] = {255, 255, 200}; // светло-желтый фон
    unsigned char headColor[3] = {255, 165, 0};  // оранжевый для головы
    unsigned char earColor[3] = {200, 100, 0};   // темно-оранжевый для ушей
    unsigned char innerEarColor[3] = {255, 192, 203}; // розовый для внутренней части ушей
    unsigned char eyeColor[3] = {0, 200, 0};     // зеленые глаза
    unsigned char pupilColor[3] = {0, 0, 0};     // черные зрачки
    unsigned char noseColor[3] = {255, 150, 150}; // розовый нос
    unsigned char mouthColor[3] = {0, 0, 0};     // черный рот
    unsigned char tongueColor[3] = {255, 0, 0};  // красный язычок
    unsigned char cheekColor[3] = {255, 192, 203}; // розовые щеки
    unsigned char whiskerColor[3] = {0, 0, 0};   // черные усы
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = (y * width + x) * 4;
            
            // Начальные значения - фон
            image[index] = bgColor[0];
            image[index + 1] = bgColor[1];
            image[index + 2] = bgColor[2];
            image[index + 3] = 255;
            
            // Нормализованные координаты (от 0 до 1)
            float fx = (float)x / width;
            float fy = (float)y / height;
            
            // Голова (круг)
            float headCenterX = 0.5f;
            float headCenterY = 0.5f;
            float headRadius = 0.35f;
            float headDist = sqrt((fx - headCenterX) * (fx - headCenterX) + 
                                 (fy - headCenterY) * (fy - headCenterY));
            
            if (headDist <= headRadius) {
                image[index] = headColor[0];
                image[index + 1] = headColor[1];
                image[index + 2] = headColor[2];
            }
            
            // Уши - треугольники
            // Левое ухо
            float ear1X1 = 0.3f, ear1Y1 = 0.7f;
            float ear1X2 = 0.4f, ear1Y2 = 0.9f;
            float ear1X3 = 0.5f, ear1Y3 = 0.7f;
            
            // Правое ухо
            float ear2X1 = 0.5f, ear2Y1 = 0.7f;
            float ear2X2 = 0.6f, ear2Y2 = 0.9f;
            float ear2X3 = 0.7f, ear2Y3 = 0.7f;
            
            // Функция для проверки точки в треугольнике
            auto pointInTriangle = [](float px, float py, 
                                     float x1, float y1, float x2, float y2, float x3, float y3) {
                float d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
                float d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3);
                float d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1);
                bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
                bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
                return !(hasNeg && hasPos);
            };
            
            // Рисуем уши
            if (pointInTriangle(fx, fy, ear1X1, ear1Y1, ear1X2, ear1Y2, ear1X3, ear1Y3) ||
                pointInTriangle(fx, fy, ear2X1, ear2Y1, ear2X2, ear2Y2, ear2X3, ear2Y3)) {
                image[index] = earColor[0];
                image[index + 1] = earColor[1];
                image[index + 2] = earColor[2];
            }
            
            // Внутренняя часть ушей (меньшие треугольники)
            float innerEar1X1 = 0.35f, innerEar1Y1 = 0.72f;
            float innerEar1X2 = 0.4f, innerEar1Y2 = 0.85f;
            float innerEar1X3 = 0.45f, innerEar1Y3 = 0.72f;
            
            float innerEar2X1 = 0.55f, innerEar2Y1 = 0.72f;
            float innerEar2X2 = 0.6f, innerEar2Y2 = 0.85f;
            float innerEar2X3 = 0.65f, innerEar2Y3 = 0.72f;
            
            if (pointInTriangle(fx, fy, innerEar1X1, innerEar1Y1, innerEar1X2, innerEar1Y2, innerEar1X3, innerEar1Y3) ||
                pointInTriangle(fx, fy, innerEar2X1, innerEar2Y1, innerEar2X2, innerEar2Y2, innerEar2X3, innerEar2Y3)) {
                image[index] = innerEarColor[0];
                image[index + 1] = innerEarColor[1];
                image[index + 2] = innerEarColor[2];
            }
            
            // Глаза - овалы
            float eye1CenterX = 0.4f, eye1CenterY = 0.55f;
            float eye2CenterX = 0.6f, eye2CenterY = 0.55f;
            float eyeRadiusX = 0.06f, eyeRadiusY = 0.08f;
            
            float eye1Dist = ((fx - eye1CenterX) * (fx - eye1CenterX)) / (eyeRadiusX * eyeRadiusX) +
                            ((fy - eye1CenterY) * (fy - eye1CenterY)) / (eyeRadiusY * eyeRadiusY);
            float eye2Dist = ((fx - eye2CenterX) * (fx - eye2CenterX)) / (eyeRadiusX * eyeRadiusX) +
                            ((fy - eye2CenterY) * (fy - eye2CenterY)) / (eyeRadiusY * eyeRadiusY);
            
            if (eye1Dist <= 1.0f || eye2Dist <= 1.0f) {
                image[index] = eyeColor[0];
                image[index + 1] = eyeColor[1];
                image[index + 2] = eyeColor[2];
            }
            
            // Зрачки - круги
            float pupil1CenterX = 0.4f, pupil1CenterY = 0.55f;
            float pupil2CenterX = 0.6f, pupil2CenterY = 0.55f;
            float pupilRadius = 0.03f;
            
            float pupil1Dist = sqrt((fx - pupil1CenterX) * (fx - pupil1CenterX) + 
                                   (fy - pupil1CenterY) * (fy - pupil1CenterY));
            float pupil2Dist = sqrt((fx - pupil2CenterX) * (fx - pupil2CenterX) + 
                                   (fy - pupil2CenterY) * (fy - pupil2CenterY));
            
            if (pupil1Dist <= pupilRadius || pupil2Dist <= pupilRadius) {
                image[index] = pupilColor[0];
                image[index + 1] = pupilColor[1];
                image[index + 2] = pupilColor[2];
            }
            
            // Блеск в глазах - маленькие белые круги
            float sparkle1X = 0.38f, sparkle1Y = 0.53f;
            float sparkle2X = 0.58f, sparkle2Y = 0.53f;
            float sparkleRadius = 0.015f;
            
            float sparkle1Dist = sqrt((fx - sparkle1X) * (fx - sparkle1X) + (fy - sparkle1Y) * (fy - sparkle1Y));
            float sparkle2Dist = sqrt((fx - sparkle2X) * (fx - sparkle2X) + (fy - sparkle2Y) * (fy - sparkle2Y));
            
            if (sparkle1Dist <= sparkleRadius || sparkle2Dist <= sparkleRadius) {
                image[index] = 255;
                image[index + 1] = 255;
                image[index + 2] = 255;
            }
            
            // Нос - треугольник
            float noseX1 = 0.47f, noseY1 = 0.48f;
            float noseX2 = 0.53f, noseY2 = 0.48f;
            float noseX3 = 0.5f, noseY3 = 0.45f;
            
            if (pointInTriangle(fx, fy, noseX1, noseY1, noseX2, noseY2, noseX3, noseY3)) {
                image[index] = noseColor[0];
                image[index + 1] = noseColor[1];
                image[index + 2] = noseColor[2];
            }
            
            // Улыбка - дуга (полукруг)
            float smileCenterX = 0.5f, smileCenterY = 0.35f;
            float smileRadius = 0.15f;
            float smileWidth = 0.02f;
            
            float smileDist = sqrt((fx - smileCenterX) * (fx - smileCenterX) + 
                                  (fy - smileCenterY) * (fy - smileCenterY));
            
            // Угол для определения положения на дуге
            float angle = atan2(fy - smileCenterY, fx - smileCenterX);
            
            // Улыбка - это дуга в нижней части круга
            if (smileDist >= smileRadius - smileWidth && smileDist <= smileRadius + smileWidth &&
                angle >= -3.14f && angle <= 0.0f) {
                image[index] = mouthColor[0];
                image[index + 1] = mouthColor[1];
                image[index + 2] = mouthColor[2];
            }
            
            // Язычок - полукруг под улыбкой
            float tongueCenterX = 0.5f, tongueCenterY = 0.3f;
            float tongueRadius = 0.05f;
            
            float tongueDist = sqrt((fx - tongueCenterX) * (fx - tongueCenterX) + 
                                   (fy - tongueCenterY) * (fy - tongueCenterY));
            
            // Язычок - это нижняя половина круга
            if (tongueDist <= tongueRadius && fy >= tongueCenterY) {
                image[index] = tongueColor[0];
                image[index + 1] = tongueColor[1];
                image[index + 2] = tongueColor[2];
            }
            
            // Щеки - круги
            float cheek1CenterX = 0.3f, cheek1CenterY = 0.5f;
            float cheek2CenterX = 0.7f, cheek2CenterY = 0.5f;
            float cheekRadius = 0.08f;
            
            float cheek1Dist = sqrt((fx - cheek1CenterX) * (fx - cheek1CenterX) + 
                                   (fy - cheek1CenterY) * (fy - cheek1CenterY));
            float cheek2Dist = sqrt((fx - cheek2CenterX) * (fx - cheek2CenterX) + 
                                   (fy - cheek2CenterY) * (fy - cheek2CenterY));
            
            if (cheek1Dist <= cheekRadius || cheek2Dist <= cheekRadius) {
                // Смешиваем цвет щек с основным цветом для прозрачности
                float blend = 0.7f;
                image[index] = (unsigned char)(image[index] * (1 - blend) + cheekColor[0] * blend);
                image[index + 1] = (unsigned char)(image[index + 1] * (1 - blend) + cheekColor[1] * blend);
                image[index + 2] = (unsigned char)(image[index + 2] * (1 - blend) + cheekColor[2] * blend);
            }
            
            // Усы - линии
            auto drawLine = [&](float x1, float y1, float x2, float y2, float thickness) {
                float lineLength = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
                float t = ((fx - x1) * (x2 - x1) + (fy - y1) * (y2 - y1)) / (lineLength * lineLength);
                t = std::max(0.0f, std::min(1.0f, t));
                
                float closestX = x1 + t * (x2 - x1);
                float closestY = y1 + t * (y2 - y1);
                
                float dist = sqrt((fx - closestX) * (fx - closestX) + (fy - closestY) * (fy - closestY));
                
                if (dist <= thickness) {
                    image[index] = whiskerColor[0];
                    image[index + 1] = whiskerColor[1];
                    image[index + 2] = whiskerColor[2];
                }
            };
            
            // Рисуем усы (6 линий)
            drawLine(0.5f, 0.45f, 0.2f, 0.4f, 0.008f);  // Левый верхний
            drawLine(0.5f, 0.45f, 0.2f, 0.45f, 0.008f); // Левый средний
            drawLine(0.5f, 0.45f, 0.2f, 0.5f, 0.008f);  // Левый нижний
            
            drawLine(0.5f, 0.45f, 0.8f, 0.4f, 0.008f);  // Правый верхний
            drawLine(0.5f, 0.45f, 0.8f, 0.45f, 0.008f); // Правый средний
            drawLine(0.5f, 0.45f, 0.8f, 0.5f, 0.008f);  // Правый нижний
        }
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    
    return texture;
}

// Функции для матриц (остаются без изменений)
void createPerspectiveProjection(float* matrix, float fov, float aspect, float near, float far) {
    float f = 1.0f / tan(fov * 0.5f);
    float range = near - far;
    
    matrix[0] = f / aspect; matrix[1] = 0.0f; matrix[2] = 0.0f; matrix[3] = 0.0f;
    matrix[4] = 0.0f; matrix[5] = f; matrix[6] = 0.0f; matrix[7] = 0.0f;
    matrix[8] = 0.0f; matrix[9] = 0.0f; matrix[10] = (far + near) / range; matrix[11] = -1.0f;
    matrix[12] = 0.0f; matrix[13] = 0.0f; matrix[14] = (2.0f * far * near) / range; matrix[15] = 0.0f;
}

void createViewMatrix(float* matrix, float eyeX, float eyeY, float eyeZ, 
                      float centerX, float centerY, float centerZ, 
                      float upX, float upY, float upZ) {
    float forward[3] = {centerX - eyeX, centerY - eyeY, centerZ - eyeZ};
    float length = sqrt(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
    forward[0] /= length; forward[1] /= length; forward[2] /= length;
    
    float right[3] = {
        forward[1] * upZ - forward[2] * upY,
        forward[2] * upX - forward[0] * upZ,
        forward[0] * upY - forward[1] * upX
    };
    length = sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    right[0] /= length; right[1] /= length; right[2] /= length;
    
    float up[3] = {
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0]
    };
    
    matrix[0] = right[0]; matrix[1] = up[0]; matrix[2] = -forward[0]; matrix[3] = 0.0f;
    matrix[4] = right[1]; matrix[5] = up[1]; matrix[6] = -forward[1]; matrix[7] = 0.0f;
    matrix[8] = right[2]; matrix[9] = up[2]; matrix[10] = -forward[2]; matrix[11] = 0.0f;
    matrix[12] = -right[0]*eyeX - right[1]*eyeY - right[2]*eyeZ;
    matrix[13] = -up[0]*eyeX - up[1]*eyeY - up[2]*eyeZ;
    matrix[14] = forward[0]*eyeX + forward[1]*eyeY + forward[2]*eyeZ;
    matrix[15] = 1.0f;
}

void createModelMatrix(float* matrix, float rotationX, float rotationY, float rotationZ) {
    float cosX = cos(rotationX), sinX = sin(rotationX);
    float rotX[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, cosX, -sinX, 0.0f,
        0.0f, sinX, cosX, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    float cosY = cos(rotationY), sinY = sin(rotationY);
    float rotY[16] = {
        cosY, 0.0f, sinY, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        -sinY, 0.0f, cosY, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    float cosZ = cos(rotationZ), sinZ = sin(rotationZ);
    float rotZ[16] = {
        cosZ, -sinZ, 0.0f, 0.0f,
        sinZ, cosZ, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    for (int i = 0; i < 16; i++) matrix[i] = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                matrix[i*4 + j] += rotZ[i*4 + k] * rotY[k*4 + j];
            }
        }
    }
    
    float temp[16];
    for (int i = 0; i < 16; i++) temp[i] = matrix[i];
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matrix[i*4 + j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                matrix[i*4 + j] += temp[i*4 + k] * rotX[k*4 + j];
            }
        }
    }
}

int main() {
    // Инициализация GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Настройка GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Создание окна
    GLFWwindow* window = glfwCreateWindow(800, 600, "3D Engine - Happy Cat Cube", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Инициализация GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // Включение теста глубины
    glEnable(GL_DEPTH_TEST);

    // Создание шейдерной программы
    unsigned int shaderProgram = createShaderProgram();

    // Создание текстуры веселого кота
    unsigned int catTexture = createHappyCatTexture();

    // Вершины куба с текстурными координатами
    float vertices[] = {
        // Позиции (x, y, z)    // Текстурные координаты (s, t)
        // Задняя грань
        -0.5f, -0.5f, -0.5f,    0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,    1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,    1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,    0.0f, 1.0f,
        
        // Передняя грань (с котом)
        -0.5f, -0.5f,  0.5f,    0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,    1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,    1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,    0.0f, 1.0f,
        
        // Левая грань
        -0.5f, -0.5f, -0.5f,    0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,    1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,    1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,    0.0f, 1.0f,
        
        // Правая грань
         0.5f, -0.5f, -0.5f,    0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,    1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,    1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,    0.0f, 1.0f,
        
        // Нижняя грань
        -0.5f, -0.5f, -0.5f,    0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,    1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,    1.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,    0.0f, 1.0f,
        
        // Верхняя грань
        -0.5f,  0.5f, -0.5f,    0.0f, 0.0f,
         0.5f,  0.5f, -0.5f,    1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,    1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,    0.0f, 1.0f
    };

    // Индексы для отрисовки треугольников
    unsigned int indices[] = {
        // Задняя грань
        0, 1, 2, 2, 3, 0,
        // Передняя грань
        4, 5, 6, 6, 7, 4,
        // Левая грань
        8, 9, 10, 10, 11, 8,
        // Правая грань
        12, 13, 14, 14, 15, 12,
        // Нижняя грань
        16, 17, 18, 18, 19, 16,
        // Верхняя грань
        20, 21, 22, 22, 23, 20
    };

    // Создание VAO, VBO, EBO
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Настройка VAO
    glBindVertexArray(VAO);

    // Копирование вершин в VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Копирование индексов в EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Настройка атрибутов вершин
    // Позиции
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Текстурные координаты
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Отвязываем буферы
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Цвет фона - яркий голубой для веселого настроения
    glClearColor(0.4f, 0.7f, 1.0f, 1.0f);

    // Получаем location uniform-переменных
    int modelLocation = glGetUniformLocation(shaderProgram, "model");
    int viewLocation = glGetUniformLocation(shaderProgram, "view");
    int projectionLocation = glGetUniformLocation(shaderProgram, "projection");
    int faceTypeLocation = glGetUniformLocation(shaderProgram, "faceType");
    int timeLocation = glGetUniformLocation(shaderProgram, "time");

    // Главный цикл
    while (!glfwWindowShouldClose(window)) {
        // Очистка экрана и буфера глубины
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Используем шейдерную программу
        glUseProgram(shaderProgram);

        // Активируем текстуру
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, catTexture);

        // Создаем матрицы
        float model[16], view[16], projection[16];
        
        // Вращение куба вокруг нескольких осей
        float time = glfwGetTime();
        createModelMatrix(model, time * 0.5f, time * 0.8f, time * 0.3f);
        
        // Камера смотрит на куб с небольшого расстояния
        createViewMatrix(view, 2.0f, 2.0f, 2.0f,  // позиция камеры
                        0.0f, 0.0f, 0.0f,        // смотрит в центр
                        0.0f, 1.0f, 0.0f);       // вектор "вверх"
        
        // Перспективная проекция
        createPerspectiveProjection(projection, 45.0f * 3.14159f / 180.0f, 800.0f/600.0f, 0.1f, 100.0f);

        // Передаем матрицы и время в шейдер
        glUniformMatrix4fv(modelLocation, 1, GL_FALSE, model);
        glUniformMatrix4fv(viewLocation, 1, GL_FALSE, view);
        glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, projection);
        glUniform1f(timeLocation, time);

        // Рисуем куб по одной грани за раз
        glBindVertexArray(VAO);
        
        // Рисуем каждую грань отдельно, чтобы задать правильный faceType
        for (int i = 0; i < 6; i++) {
            if (i == 1) { // Передняя грань - с котом
                glUniform1i(faceTypeLocation, 1);
            } else { // Остальные грани - цветные
                glUniform1i(faceTypeLocation, 0);
            }
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)(i * 6 * sizeof(unsigned int)));
        }

        // Обмен буферов и обработка событий
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Очистка ресурсов
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &catTexture);

    glfwTerminate();
    return 0;
}