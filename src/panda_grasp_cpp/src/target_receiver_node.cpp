#include <memory>
#include <functional>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"


class TargetReceiverNode : public rclcpp::Node
{
public:
    TargetReceiverNode() : Node("target_receiver_cpp")
    {
        // 订阅 Python 视觉节点发布的目标坐标
        subscription_ =
            this->create_subscription<
                geometry_msgs::msg::PointStamped>(
                "/grasp/target_point",
                10,
                std::bind(
                    &TargetReceiverNode::target_callback,
                    this,
                    std::placeholders::_1
                )
            );

        RCLCPP_INFO(
            get_logger(),
            "C++ target receiver started"
        );
    }

private:
    // 目标坐标回调函数
    void target_callback(
        const geometry_msgs::msg::PointStamped::SharedPtr msg)
    {
        // 1. 获取目标坐标
        double x = msg->point.x;
        double y = msg->point.y;
        double z = msg->point.z;

        // 2. 检查是否存在无效数值
        if (!std::isfinite(x) ||
            !std::isfinite(y) ||
            !std::isfinite(z))
        {
            RCLCPP_WARN(
                get_logger(),
                "Invalid target coordinates"
            );
            return;
        }

        // 3. 输出坐标信息
        // 最多每1000毫秒输出一次，避免刷屏
        RCLCPP_INFO_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "Target [%s]: X=%.3f Y=%.3f Z=%.3f m",
            msg->header.frame_id.c_str(),
            x, y, z
        );
    }

    // 订阅者成员变量
    rclcpp::Subscription<
        geometry_msgs::msg::PointStamped>::SharedPtr
        subscription_;
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<TargetReceiverNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}