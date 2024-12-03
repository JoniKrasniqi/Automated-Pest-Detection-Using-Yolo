# 🐞 Automated Pest Detection System
An AI-powered application that utilizes **YOLOv5** for real-time pest detection in agricultural images, integrated into an interactive web application using **Streamlit**. This system assists farmers and agricultural professionals in early pest detection, promoting sustainable farming practices by reducing crop losses and minimizing pesticide usage.

## Table of Contents
  - Features
  - Demo
  - Installation
  - Contributing
  - License
  - Acknowledgments

## Features
- Real-time Pest Detection: Upload images and get instant detection results with annotated bounding boxes and labels.
- Comprehensive Pest Library: Detects ten common agricultural pests:
  - Armyworm
  - Legume Blister Beetle
  - Red Spider
  - Rice Gall Midge
  - Rice Leaf Roller
  - Rice Leafhopper
  - Rice Water Weevil
  - Wheat Phloeothrips
  - White Backed Plant Hopper
  - Yellow Rice Borer
- Analytics Dashboard: Monitor pest detection trends over time, including most frequently detected pests and detection history.
- User-friendly Interface: Intuitive design with easy navigation and responsive layout.
- Model Integration: Utilizes a custom-trained YOLOv5 model for accurate and efficient pest detection.

## Demo
### Streamlit application
https://pestdetector.streamlit.app/
##### 1. First upload an image that contains one of the pests mentioned
<img width="958" alt="uploadimage" src="https://github.com/user-attachments/assets/34610b72-6d27-4223-9f43-0e1f771a43c6">

##### 2. After uploading click run detection
<img width="960" alt="uploadedimage" src="https://github.com/user-attachments/assets/1f3996b3-5936-4b19-981f-d4e9066feba9">

##### 3. The result with the pest detected and confidence level
<img width="960" alt="detectioncomplete" src="https://github.com/user-attachments/assets/82eb6766-cf2a-4ba9-9d37-0a190270f2d5">

## Installation
### Prerequisites
      - Python 3.7 or higher
      - Git
### Cloning the Repository

      git clone https://github.com/JoniKrasniqi/Automated-Pest-Detection-Using-Yolo
      cd Automated-Pest-Detection-Using-Yolo

### Installing Dependencies
##### Python Dependencies
        Install the required Python packages using pip:
        pip install -r requirements.txt

### Run the app locally
        streamlit run main.py


## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the **MIT** License.

## Acknowledgments

- This project utilizes the [YOLOv5](https://github.com/ultralytics/yolov5) object detection algorithm developed by [Ultralytics](https://ultralytics.com). We thank the developers for their valuable contribution to the open-source community.

  If you use **YOLOv5** in your research or development, please cite it as follows:

  ```bibtex
  @software{yolov5,
    title = {Ultralytics YOLOv5},
    author = {Glenn Jocher},
    year = {2020},
    version = {7.0},
    license = {AGPL-3.0},
    url = {https://github.com/ultralytics/yolov5},
    doi = {10.5281/zenodo.3908559},
    orcid = {0000-0001-5950-6979}
  }
- **Streamlit**: For enabling rapid development of interactive web applications.
- **Contributors**: Everyone who has contributed to improving this project.

## Future Work
  - Expand Pest Library: Incorporate additional pest species to enhance the system's utility.
  - Mobile Deployment: Develop a mobile application for field use by farmers.
  - Integration with IoT Devices: Connect with drones or cameras for continuous monitoring.
  - Predictive Analytics: Incorporate environmental data to predict pest infestations.
