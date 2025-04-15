#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onSelectModel();
    void onSelectImage();
    void detectObjects();

private:
    void loadAndDetectYolo();
    cv::Mat processImage(const cv::Mat& frame, cv::dnn::Net& net);
    
    QLabel* imageLabel;
    QPushButton* selectModelButton;
    QPushButton* selectImageButton;
    std::vector<std::string> classes;
    QString modelPath;
    QString imagePath;
};
#endif // MAINWINDOW_H
