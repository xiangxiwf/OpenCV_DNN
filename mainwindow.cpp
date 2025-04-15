#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // 创建主窗口部件和布局
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);
    setCentralWidget(centralWidget);
    
    // 创建按钮布局
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    
    // 创建选择按钮
    selectModelButton = new QPushButton("选择模型", this);
    selectImageButton = new QPushButton("选择图片", this);
    
    // 添加按钮到布局
    buttonLayout->addWidget(selectModelButton);
    buttonLayout->addWidget(selectImageButton);
    
    // 创建图像标签
    imageLabel = new QLabel(this);
    imageLabel->setMinimumSize(640, 480);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setText("请选择模型和图片");
    
    // 添加所有控件到主布局
    mainLayout->addLayout(buttonLayout);
    mainLayout->addWidget(imageLabel);
    
    // 设置窗口属性
    setWindowTitle("YOLOv11 目标检测");
    resize(800, 600);
    
    // 加载COCO数据集类别
    std::string names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    classes = std::vector<std::string>(std::begin(names), std::end(names));
    
    // 连接信号和槽
    connect(selectModelButton, &QPushButton::clicked, this, &MainWindow::onSelectModel);
    connect(selectImageButton, &QPushButton::clicked, this, &MainWindow::onSelectImage);
}

MainWindow::~MainWindow() {}

void MainWindow::onSelectModel()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("选择ONNX模型文件"), "", tr("ONNX Files (*.onnx)"));
    
    if (!fileName.isEmpty()) {
        modelPath = fileName;
        if (!imagePath.isEmpty()) {
            detectObjects();
        }
    }
}

void MainWindow::onSelectImage()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("选择图片"), "", tr("Images (*.png *.jpg *.jpeg *.bmp)"));
    
    if (!fileName.isEmpty()) {
        imagePath = fileName;
        if (!modelPath.isEmpty()) {
            detectObjects();
        }
    }
}

void MainWindow::detectObjects()
{
    if (modelPath.isEmpty() || imagePath.isEmpty()) {
        imageLabel->setText("请先选择模型和图片");
        return;
    }
    loadAndDetectYolo();
}

void MainWindow::loadAndDetectYolo()
{
    try {
        // 加载模型
        cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath.toStdString());
        
        // 设置计算后端
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // 读取图像
        cv::Mat frame = cv::imread(imagePath.toStdString());
        
        if (frame.empty()) {
            qDebug() << "无法读取图像: " << imagePath;
            imageLabel->setText("无法读取图像!");
            return;
        }
        
        qDebug() << "图像加载成功，尺寸：" << frame.cols << "x" << frame.rows;
        
        // 处理图像并显示结果
        cv::Mat resultImage = processImage(frame, net);
        
        // 将OpenCV的Mat转换为Qt的QImage
        QImage qimg(resultImage.data, resultImage.cols, resultImage.rows, 
                    static_cast<int>(resultImage.step), QImage::Format_RGB888);
        qimg = qimg.rgbSwapped(); // BGR to RGB
        
        // 显示结果
        imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(
                             imageLabel->size(), 
                             Qt::KeepAspectRatio, 
                             Qt::SmoothTransformation));
        
    } catch (const cv::Exception& e) {
        qDebug() << "OpenCV错误: " << e.what();
        imageLabel->setText("处理图像时发生错误!");
    } catch (const std::exception& e) {
        qDebug() << "标准错误: " << e.what();
        imageLabel->setText("处理图像时发生错误!");
    }
}

cv::Mat MainWindow::processImage(const cv::Mat& frame, cv::dnn::Net& net)
{
    cv::Mat blob;
    cv::Mat resultImage = frame.clone();
    
    // 预处理 - 图像转换为blob
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    
    // 设置网络输入
    net.setInput(blob);
    
    // 前向传播
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    qDebug() << "网络输出层数: " << outputs.size();
    for (size_t i = 0; i < outputs.size(); i++) {
        qDebug() << "输出" << i << "形状: " << outputs[i].dims << "维";
        for (int j = 0; j < outputs[i].dims; j++) {
            qDebug() << "  维度" << j << ": " << outputs[i].size[j];
        }
    }
    
    if (outputs.empty()) {
        qDebug() << "网络输出为空!";
        return resultImage;
    }
    
    // 处理YOLOv11输出
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // 获取输入图像的原始尺寸，用于坐标转换
    float scale_x = static_cast<float>(frame.cols) / 640.0f;
    float scale_y = static_cast<float>(frame.rows) / 640.0f;
    
    // 从输出日志看，输出格式是 [1, 84, 8400]
    // 这可能是转置的YOLOv8输出，其中：
    // - 84 是 4(坐标) + 80(类别)
    // - 8400 是候选框的数量
    if (outputs[0].dims == 3 && outputs[0].size[1] == 84) {
        qDebug() << "使用转置的YOLOv8格式解析";
        
        const int rows = outputs[0].size[2]; // 8400个框
        const int dimensions = outputs[0].size[1]; // 84个参数
        const int num_classes = dimensions - 4; // 假设前4个是坐标，后面都是类别
        
        // 获取输出数据
        cv::Mat output = outputs[0].reshape(1, dimensions); // 调整为合适的形状
        
        // 遍历所有框
        for (int i = 0; i < rows; ++i) {
            // 寻找最高类别置信度
            float maxClassScore = 0;
            int maxClassId = 0;
            
            for (int c = 0; c < num_classes; c++) {
                float score = output.at<float>(c + 4, i);
                if (score > maxClassScore) {
                    maxClassScore = score;
                    maxClassId = c;
                }
            }
            
            // 阈值过滤
            if (maxClassScore > 0.25) { // 降低阈值以检测更多目标
                // 解析边界框坐标 (前4个值)
                float x = output.at<float>(0, i);
                float y = output.at<float>(1, i);
                float w = output.at<float>(2, i);
                float h = output.at<float>(3, i);
                
                qDebug() << "原始坐标:" << x << y << w << h;
                
                // 修改后的坐标转换逻辑，考虑输入图像的缩放
                int left = static_cast<int>((x - w/2) * scale_x);
                int top = static_cast<int>((y - h/2) * scale_y);
                int width = static_cast<int>(w * scale_x);
                int height = static_cast<int>(h * scale_y);
                
                // 确保框在图像范围内
                left = std::max(0, std::min(left, frame.cols - 1));
                top = std::max(0, std::min(top, frame.rows - 1));
                width = std::min(width, frame.cols - left);
                height = std::min(height, frame.rows - top);
                
                // 添加有效框
                if (width > 0 && height > 0) {
                    boxes.push_back(cv::Rect(left, top, width, height));
                    classIds.push_back(maxClassId);
                    confidences.push_back(maxClassScore);
                    
                    qDebug() << "检测到对象: 类别=" << maxClassId << ", 置信度=" << maxClassScore 
                             << ", 位置=" << left << "," << top << "," << width << "," << height;
                }
            }
        }
    }
    // 尝试使用标准YOLOv8输出格式 [batch, detection, 85]
    else if (outputs[0].dims == 3 && outputs[0].size[2] > 4) {
        // 标准YOLOv5-v8格式
        const int rows = outputs[0].size[1];
        const int dimensions = outputs[0].size[2];
        
        qDebug() << "使用标准YOLOv5-v8格式解析，检测到" << rows << "个候选框";
        
        float* data = (float*)outputs[0].data;
        for (int i = 0; i < rows; ++i) {
            float* row = data + i * dimensions;
            
            // 获取类别得分
            int classId = 0;
            float maxScore = 0;
            
            // 从第5个元素开始是类别得分
            for (int j = 5; j < dimensions && j < 85; j++) {
                if (row[j] > maxScore) {
                    maxScore = row[j];
                    classId = j - 5;
                }
            }
            
            float confidence = row[4] * maxScore; // objectness * class confidence
            
            // 筛选掉低置信度的框
            if (confidence > 0.3) {
                float cx = row[0];
                float cy = row[1];
                float width = row[2];
                float height = row[3];
                
                // 转换为图像坐标
                int left = static_cast<int>((cx - width/2) * frame.cols);
                int top = static_cast<int>((cy - height/2) * frame.rows);
                int boxWidth = static_cast<int>(width * frame.cols);
                int boxHeight = static_cast<int>(height * frame.rows);
                
                // 确保框在图像范围内
                left = std::max(0, std::min(left, frame.cols - 1));
                top = std::max(0, std::min(top, frame.rows - 1));
                boxWidth = std::min(boxWidth, frame.cols - left);
                boxHeight = std::min(boxHeight, frame.rows - top);
                
                // 添加有效框
                if (boxWidth > 0 && boxHeight > 0) {
                    boxes.push_back(cv::Rect(left, top, boxWidth, boxHeight));
                    classIds.push_back(classId);
                    confidences.push_back(confidence);
                    
                    qDebug() << "检测到对象: 类别=" << classId << ", 置信度=" << confidence 
                             << ", 位置=" << left << "," << top << "," << boxWidth << "," << boxHeight;
                }
            }
        }
    }
    else {
        qDebug() << "使用通用格式解析";
        // 输出一些原始数据用于调试
        if (outputs[0].dims >= 2) {
            qDebug() << "数据样本:";
            
            // 设置最大安全访问范围
            int maxSamples = std::min(5, outputs[0].size[0]);
            int maxValues = 10;
            
            if (outputs[0].dims >= 3) {
                for (int i = 0; i < std::min(3, outputs[0].size[1]); i++) {
                    QString values;
                    for (int j = 0; j < std::min(maxValues, outputs[0].size[2]); j++) {
                        values += QString::number(outputs[0].at<float>(0, i, j)) + " ";
                    }
                    qDebug() << "样本" << i << ":" << values;
                }
            }
            else {
                // 2D输出
                for (int i = 0; i < maxSamples; i++) {
                    QString values;
                    for (int j = 0; j < std::min(maxValues, outputs[0].cols); j++) {
                        values += QString::number(outputs[0].at<float>(i, j)) + " ";
                    }
                    qDebug() << "样本" << i << ":" << values;
                }
            }
        }
        
        // 在图像上显示提示
        cv::putText(resultImage, "Unsupported model output format", cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    
    qDebug() << "检测到" << boxes.size() << "个有效边界框";
    
    if (!boxes.empty()) {
        // 非极大值抑制，去除重叠框
        std::vector<int> indices;
        try {
            cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indices);
            qDebug() << "NMS后保留" << indices.size() << "个边界框";
            
            // 确保类别索引在有效范围内
            for (size_t i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                if (idx >= 0 && idx < boxes.size()) {
                    cv::Rect box = boxes[idx];
                    int classId = classIds[idx];
                    
                    // 确保类别ID在有效范围内
                    if (classId >= 0 && classId < static_cast<int>(classes.size())) {
                        // 计算线条粗细和字体大小的缩放因子
                        float scale_factor = std::min(frame.cols, frame.rows) / 500.0f;
                        int thickness = std::max(1, static_cast<int>(scale_factor * 2));
                        float font_scale = std::max(0.5f, scale_factor * 0.6f);
                        
                        // 使用动态计算的线条粗细
                        cv::rectangle(resultImage, box, cv::Scalar(0, 255, 0), thickness);
                        
                        // 调整标签显示
                        std::string label = classes[classId] + ": " + std::to_string(static_cast<int>(confidences[idx] * 100)) + "%";
                        int baseLine;
                        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseLine);
                        
                        // 设置标签背景和文字
                        int y = std::max(box.y, labelSize.height);
                        cv::rectangle(resultImage, cv::Point(box.x, y - labelSize.height),
                                    cv::Point(box.x + labelSize.width, y + baseLine),
                                    cv::Scalar(0, 255, 0), cv::FILLED);
                        cv::putText(resultImage, label, cv::Point(box.x, y),
                                  cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness);
                        
                        qDebug() << "绘制框: " << box.x << "," << box.y << "," << box.width << "," << box.height
                                 << " 类别: " << QString::fromStdString(classes[classId])
                                 << " 置信度: " << confidences[idx];
                    } else {
                        qDebug() << "类别ID超出范围: " << classId;
                    }
                } else {
                    qDebug() << "索引超出范围: " << idx;
                }
            }
        } catch (const std::exception& e) {
            qDebug() << "NMS处理错误: " << e.what();
            cv::putText(resultImage, "Error in detection processing", cv::Point(50, 50), 
                      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }
    } else {
        qDebug() << "未检测到任何对象，请检查模型输出格式";
        cv::putText(resultImage, "No objects detected", cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    
    return resultImage;
}
