import matplotlib.pyplot as plt

# Data for plotting
models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

# Video data
video_speed = [102.85, 169.88, 324.7, 507.1, 712.8]
video_avg_detection_car = [0.690865334, 0.693776712, 0.733719195, 0.745773162, 0.742526069]
video_avg_detection_truck = [0.642739951, 0.692430797, 0.741093959, 0.771655914, 0.7817712]
video_count_car = [11407, 11369, 15217, 17220, 18515]
video_count_truck = [546, 2081, 2211, 2636, 3265]

# Image data
image_speed = [0.36, 0.52, 0.82, 1.27, 1.64]
image_avg_detection_car = [0.751728385, 0.78182325, 0.788812438, 0.8398252, 0.826808133]
image_count_car = [13, 16, 16, 15, 15]

# 1. Результаты обработки видео по классу "car"
fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.set_xlabel('Model')
ax1.set_ylabel('Processing Speed (video fps)', color='tab:blue')
ax1.plot(models, video_speed, marker='o', linestyle='-', color='tab:blue', label='Video Speed')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Average Detection Percentage', color='tab:red')
ax2.plot(models, video_avg_detection_car, marker='x', linestyle='-', color='tab:red', label='Video Car Detection %')
ax2.tick_params(axis='y', labelcolor='tab:red')
fig1.tight_layout()
fig1.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Video Processing Results for Class "car"')
fig1.savefig("video_processing_results_car.png")
plt.close(fig1)

# 2. Результаты обработки видео по классу "truck"
fig2, ax1 = plt.subplots(figsize=(14, 8))
ax1.set_xlabel('Model')
ax1.set_ylabel('Processing Speed (video fps)', color='tab:blue')
ax1.plot(models, video_speed, marker='o', linestyle='-', color='tab:blue', label='Video Speed')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Average Detection Percentage', color='tab:red')
ax2.plot(models, video_avg_detection_truck, marker='x', linestyle='-', color='tab:red', label='Video Truck Detection %')
ax2.tick_params(axis='y', labelcolor='tab:red')
fig2.tight_layout()
fig2.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Video Processing Results for Class "truck"')
fig2.savefig("video_processing_results_truck.png")
plt.close(fig2)

# 3. Результаты обработки видео по обоим классам
fig3, ax1 = plt.subplots(figsize=(14, 8))
ax1.set_xlabel('Model')
ax1.set_ylabel('Processing Speed (video fps)', color='tab:blue')
ax1.plot(models, video_speed, marker='o', linestyle='-', color='tab:blue', label='Video Speed')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Average Detection Percentage', color='tab:red')
ax2.plot(models, video_avg_detection_car, marker='x', linestyle='-', color='tab:red', label='Video Car Detection %')
ax2.plot(models, video_avg_detection_truck, marker='x', linestyle='--', color='tab:red', label='Video Truck Detection %')
ax2.tick_params(axis='y', labelcolor='tab:red')
fig3.tight_layout()
fig3.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Video Processing Results for Both Classes')
fig3.savefig("video_processing_results_both.png")
plt.close(fig3)

# 4. Результаты обработки изображения
fig4, ax1 = plt.subplots(figsize=(14, 8))
ax1.set_xlabel('Model')
ax1.set_ylabel('Processing Speed (image s)', color='tab:blue')
ax1.plot(models, image_speed, marker='o', linestyle='--', color='tab:blue', label='Image Speed')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Average Detection Percentage', color='tab:red')
ax2.plot(models, image_avg_detection_car, marker='s', linestyle='-', color='tab:red', label='Image Car Detection %')
ax2.tick_params(axis='y', labelcolor='tab:red')
fig4.tight_layout()
fig4.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.title('Image Processing Results')
fig4.savefig("image_processing_results.png")
plt.close(fig4)
