import os
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# 1. 这里填你截图里显示的实际 MCAP 文件路径
MCAP_PATH = "dataset/my_experiment_data/single_chip/adc_samples/data/2025_12_09_drone_mmwave_1_1_0.mcap_0.mcap"

def inspect_mcap():
    print(f"Checking file: {MCAP_PATH}")
    
    if not os.path.exists(MCAP_PATH):
        print("[ERROR] 文件不存在！请检查路径是否正确。")
        print(f"当前工作目录: {os.getcwd()}")
        return

    topic_counts = {}
    
    try:
        with open(MCAP_PATH, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            print("[INFO] 文件打开成功，正在扫描 Topic...")
            
            # 遍历所有 schema 和 channel 来获取 topic 信息
            summary = reader.get_summary()
            if summary and summary.statistics:
                print("[INFO] 读取到统计信息 (Summary):")
                for ch_id, count in summary.statistics.channel_message_counts.items():
                    channel = summary.channels[ch_id]
                    print(f"  - Topic: {channel.topic:<30} | Count: {count}")
                    topic_counts[channel.topic] = count
            else:
                print("[WARN] 无统计信息，开始全量扫描...")
                for schema, channel, message in reader.iter_messages():
                    topic_counts[channel.topic] = topic_counts.get(channel.topic, 0) + 1

    except Exception as e:
        print(f"[ERROR] 读取 MCAP 失败: {e}")
        return

    print("\n" + "="*40)
    print("扫描结果汇总:")
    print("="*40)
    
    if not topic_counts:
        print("未发现任何 Topic！文件可能为空或损坏。")
    else:
        for topic, count in topic_counts.items():
            print(f"Topic: {topic} \t 数量: {count} 帧")
            
            # 检查是否包含 Raw Data 关键字
            if "raw" in topic or "radar" in topic or "scan" in topic:
                print(f"  >>> 推荐: 请在 core/record.py 中将 target_topic 修改为 '{topic}'")

if __name__ == "__main__":
    inspect_mcap()