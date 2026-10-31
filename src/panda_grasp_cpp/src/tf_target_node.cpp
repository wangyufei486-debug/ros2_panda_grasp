#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

#include "tf2/exceptions.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using PointStamped = geometry_msgs::msg::PointStamped;

class TFTargetNode : public rclcpp::Node
{
public:
    TFTargetNode() : Node("tf_target_cpp")
    {
        // 1. 创建 TF Buffer
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(
            get_clock()
        );

        // 2. 创建 TF Listener
        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(
                *tf_buffer_
            );

        // 3. 订阅相机坐标
        subscription_ = create_subscription<PointStamped>(
            "/grasp/target_point_camera",
            10,
            [this](PointStamped::SharedPtr msg)
            {
                target_callback(msg);
            }
        );
    }

private:
    void target_callback(PointStamped::SharedPtr msg)
    {
        if (msg->header.frame_id.empty())
        {
            RCLCPP_WARN(get_logger(), "Missing source frame");
            return;
        }

        try
        {
            // 查询从消息坐标系到 world 的变换
            auto transform = tf_buffer_->lookupTransform(
                "world",
                msg->header.frame_id,
                tf2::TimePointZero
            );

            // 执行坐标变换
            PointStamped world_point;

            tf2::doTransform(
                *msg,
                world_point,
                transform
            );

            // 输出世界坐标
            RCLCPP_INFO_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "World: X=%.3f Y=%.3f Z=%.3f",
                world_point.point.x,
                world_point.point.y,
                world_point.point.z
            );
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(
                get_logger(),
                "TF failed: %s",
                ex.what()
            );
        }
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

    std::shared_ptr<tf2_ros::TransformListener>
        tf_listener_;

    rclcpp::Subscription<PointStamped>::SharedPtr
        subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<TFTargetNode>());

    rclcpp::shutdown();
    return 0;
}