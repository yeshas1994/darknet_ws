#include "ros/ros.h"
#include "std_msgs/Int64.h"
#include "iostream"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
//#include "sensor_msgs/PointCloud2"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "learning_depth/send.h"
//#include <pcl_ros/point_cloud.h> 
//#include <pcl/point_types>

cv::Point depthCenter;
cv::Mat final_image;
namespace enc = sensor_msgs::image_encodings;

int low_h = 0, high_h = 179, low_s = 0, high_s = 255, low_v = 0, high_v = 255;

void trackbar(int, void*) {
// function called when trackbar value changes
}

void depthCallback(const sensor_msgs::ImageConstPtr& msg2){
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg2, enc::TYPE_16UC1); //TYPE_16UC1 for depth generally
    }

    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    //cv::Mat img, canny;
    //cv::Canny(cv_ptr->image, canny, 10,100);
    //cv::imshow("View", canny);

    //        cv::imshow("depth", cv_ptr->image);
    //        cv::waitKey(10);

    int depth = cv_ptr->image.at<short int>(::depthCenter); // cv:Point(x,y) -- location of pixel u want
    ROS_INFO("Depth: %d", depth);
}

void viewercallback(const sensor_msgs::ImageConstPtr& msg){
    cv_bridge::CvImagePtr cv_view;
    cv::Mat img_mask1, img_mask2, img_maskfinal, img_hsv;
    cv::Mat obj;

    try {
    cv_view = cv_bridge::toCvCopy(msg, enc::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("%s", e.what());
        return;
    }
    
    cv::imshow("View", cv_view->image);
    cv::waitKey(10);    
    cv::cvtColor(cv_view->image, img_hsv, CV_BGR2HSV);

    //-- Trackbars to set thresholds for HSV values
    //        cv::inRange(img_hsv, cv::Scalar(0,100,100), cv::Scalar(10,255,255), img_mask1);// lower range for red
    //        cv::inRange(img_hsv, cv::Scalar(160,100,100), cv::Scalar(179,255,255), img_mask2);//upper range for red
    //        img_maskfinal = img_mask1 | img_mask2; //combines the lower and upper range if required
    
    cv::Mat kernel; // structuring element
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)); // The rectangular structural element 

    cv::inRange(img_hsv, cv::Scalar(low_h,low_s,low_v), cv::Scalar(high_h,high_s,high_v), img_maskfinal);// range for green
    cv::Mat erosion_img, closing_img, opening_img;
    cv::erode(img_maskfinal, erosion_img, kernel);
    cv::morphologyEx(erosion_img, closing_img, cv::MORPH_CLOSE, kernel); 
    cv::morphologyEx(closing_img, img_maskfinal, cv::MORPH_OPEN, kernel);
    // binary image might still have static/background noise, use morphology to try and improve image
    /**
     * Several possible methods to improve the mask image
     * Erosion, Opening or Closing using a structuralElement 
     * Maybe a combination of two?
     */

    cv::imshow("mask", img_maskfinal);
    cv::waitKey(10);
    cv::Mat canny;

    std::vector< std::vector<cv::Point> > contour_op; 
    std::vector<cv::Vec4i> hierarchy; //required only for find contour function

    cv::Canny(img_maskfinal, canny, 10, 100); //this might be  useful in removing noise 

//    cv::imshow("canny", canny);
    //findContours used 
    cv::findContours(img_maskfinal, contour_op, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    std::vector<cv::Moments> mu(contour_op.size());
    std::vector<std::vector<cv::Point> > objekt;
    std::vector<cv::Rect> objektBox;
    for (size_t i = 0; i < contour_op.size(); i++) {
        mu[i] = cv::moments(contour_op[i], false);
        
        cv::Rect oBox;
        oBox = cv::boundingRect(contour_op[i]);
        if (oBox.area() > 1000) {
            objekt.push_back(contour_op[i]);
            objektBox.push_back(oBox);
        }
    }

    std::vector<cv::Point2f> mc(contour_op.size());

    for (size_t i = 0; i < contour_op.size(); i++)
        mc[i] = cv::Point2f( static_cast<float>(mu[i].m10/mu[i].m00), 
                static_cast<float>(mu[i].m01/mu[i].m00));

    cv::Mat draw = cv::Mat::zeros(canny.size(), CV_8UC3); 
    cv::Mat img = cv_view->image;
    cv::Mat img2 = cv_view->image;
    cv::Point center;
    for (size_t i = 0; i<objekt.size(); i++) //size_t is a type guaranteed to hold any array index.
    {
        cv::drawContours(img2, objekt, (int) i, CV_RGB(20,150,25), 1);
        cv::rectangle(img, objektBox[i], CV_RGB(0,255,0), 2);

        center.x = objektBox[i].x + objektBox[i].width / 2;
        center.y = objektBox[i].y + objektBox[i].height / 2;
        cv::circle(img, center, 2, cv::Scalar(50,100,255), 4);        
    } 
        
        ::depthCenter = center;
    //This above code seems redundant as we already have the mask filled out.   

    //for (size_t i = 0; i<contour_op.size(); i++) 
      //  cv::circle(cv_view->image, mc[i], 4, cv::Scalar(50,100,255), 4);

    //Understand mc/mu moments!!!
        
    //cv::imshow("canny", canny);
    //cv::vconcat(img, canny, final_image);
    //cv::hconcat(final_image, canny, final_image);
    //cv::imshow("View", img);    
    
    //cv::Mat mask = cv::Mat::zeros(img_maskfinal.rows + 2, img_maskfinal.cols + 2, CV_8U);
    //cv::floodFill(img_maskfinal, mask, cv::Point(0,0), 255, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    //cv::imshow("view", img_maskfinal);
    //save the masked image showing the picture of the object u are trying to detect and then calculate depth
    /*
       cv::findNonZero(img_maskfinal, obj);        
       int n = obj.total()/2; 

    // DEBUG x,y coordinates  
    // for (int i = 0; i < obj.total(); i++)

    // std::cout << "Point "<< n << ":"  << obj.at<cv::Point>(n).x << ", " << obj.at<cv::Point>(n).y <<std::endl; 
    // Below draws a circle at a particular point

    //cv::imshow("view", cv_view->image);

    ::pointX = obj.at<cv::Point>(n).x; 
    ::pointY = obj.at<cv::Point>(n).y;     
    //cv::circle(cv_view->image, cv::Point(::pointX, ::pointY), 10, cv::Scalar(0,0,255), 4); 
    //        cv::imshow("view", cv_view->image);
    cv::waitKey(5);


    ros::NodeHandle nC;
    learning_depth::send send; 
    send.x = ::pointX;
    send.y = ::pointY;
    ROS_INFO("%d, %d", ::pointX, ::pointY);
    ros::Publisher pub = nC.advertise<learning_depth::send>("/coordinates", 1);


    int count = 0;
    while(nC.ok()){ 
    pub.publish(send);
    ros::spinOnce();
    count++;
    if (count > 1) break;
    }

    if (count > 1) return;
     */

}


int main(int argc, char **argv){
    ros::init(argc, argv, "depth_calculator");
    ros::NodeHandle nh;
    cv::namedWindow("Object Detection");
    //-- Trackbars to set thresholds for RGB values
    cv::createTrackbar("Low H","Object Detection", &low_h, 179, trackbar);
    cv::createTrackbar("High H","Object Detection", &high_h, 179, trackbar);
    cv::createTrackbar("Low S","Object Detection", &low_s, 255, trackbar);
    cv::createTrackbar("High S","Object Detection", &high_s, 255, trackbar);
    cv::createTrackbar("Low V","Object Detection", &low_v, 255, trackbar);
    cv::createTrackbar("High V","Object Detection", &high_v, 255, trackbar);

    cv::waitKey(10);
    //cv::createTrackbar("Low R","Object Detection", &low_r, 255, on_low_r_thresh_trackbar);
    //cv::createTrackbar("High R","Object Detection", &high_r, 255, on_high_r_thresh_trackbar);
    //cv::createTrackbar("Low G","Object Detection", &low_g, 255, on_low_g_thresh_trackbar);
    //cv::createTrackbar("High G","Object Detection", &high_g, 255, on_high_g_thresh_trackbar);
    //cv::createTrackbar("Low B","Object Detection", &low_b, 255, on_low_b_thresh_trackbar);
    //cv::createTrackbar("High B","Object Detection", &high_b, 255, on_high_b_thresh_trackbar);

    cv::namedWindow("View");
    cv::namedWindow("mask");
    //    cv::namedWindow("canny");
    cv::startWindowThread();
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub2 = it.subscribe("usb_cam/image_raw", 1, viewercallback);
    image_transport::Subscriber sub3 = it.subscribe("/kinect2/qhd/image_depth_rect", 1, depthCallback);
    //	ros::Subscriber sub4 = nh.subscribe<PointCloud>("/kinect2/qhd/points", 1, depthCallback);
    ros::spin();

    cv::destroyWindow("view");
}


