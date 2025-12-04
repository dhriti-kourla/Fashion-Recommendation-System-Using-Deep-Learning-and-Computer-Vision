# Fashion Recommendation System Using Deep Learning and Computer Vision# Fashion Recommendation System Using Deep Learning and Computer Vision<h2 align="center">SmartStylist: A Fashion Recommender System powered by Computer Vision</h2>



An intelligent fashion recommendation system that leverages state-of-the-art deep learning techniques and computer vision to analyze clothing images and provide personalized style suggestions based on visual similarity.<br>



[Try the Demo](http://localhost:8501)An intelligent fashion recommendation system that leverages state-of-the-art deep learning techniques and computer vision to analyze clothing images and provide personalized style suggestions based on visual similarity.



## 🎯 Project Overview<center>



This system combines object detection, deep learning-based feature extraction, and efficient similarity search to deliver accurate fashion recommendations. The model analyzes uploaded fashion images, detects clothing items, extracts visual features, and retrieves the most similar items from a curated dataset.## 🎯 Project Overview<a href="https://www.joankusuma.com/post/smart-stylist-a-fashion-recommender-system-powered-by-computer-vision"><img src='https://img.shields.io/badge/Project_Page-SmartStylist-pink' alt='Project Page'></a> 



## ✨ Key Features<a href='https://www.joankusuma.com/post/object-detection-model-yolov5-on-fashion-images'><img src='https://img.shields.io/badge/Project_Page-ObjectDetection-blue' alt='Object Detection'></a> 



- **Advanced Object Detection**: Utilizes YOLOv8 for precise clothing item detection and classificationThis system combines object detection, deep learning-based feature extraction, and efficient similarity search to deliver accurate fashion recommendations. The model analyzes uploaded fashion images, detects clothing items, extracts visual features, and retrieves the most similar items from a curated dataset.<a href='https://www.joankusuma.com/post/powering-visual-search-with-image-embedding'><img src='https://img.shields.io/badge/Project_Page-VisualSearch-green'></a> 

- **Deep Feature Extraction**: Custom CNN-based featurizer model for robust visual embeddings

- **Efficient Similarity Search**: FAISS-powered vector indexing for real-time recommendations<a href='https://smartstylist.streamlit.app'><img src='https://img.shields.io/badge/Streamlit-Demo-red'></a>

- **Interactive Web Interface**: Streamlit-based UI for seamless user experience

- **Multi-Category Support**: Handles diverse clothing categories including:## ✨ Key Features</center>

  - Coats & Jackets

  - Dresses & Skirts<br>

  - Shirts & Tops

  - Pants & Shorts- **Advanced Object Detection**: Utilizes YOLOv8 for precise clothing item detection and classification<br>

  - Shoes & Accessories

  - And more...- **Deep Feature Extraction**: Custom CNN-based featurizer model for robust visual embeddings<figure>



## 🏗️ Architecture- **Efficient Similarity Search**: FAISS-powered vector indexing for real-time recommendations    <center>



### 1. Object Detection Pipeline- **Interactive Web Interface**: Streamlit-based UI for seamless user experience        <img src="https://static.wixstatic.com/media/81114d_7f499b8207b848bc8bccfe1035a28b3d~mv2.png" alt="flowchart" height="350" width="600">

- **Model**: YOLOv8 (ONNX format)

- **Purpose**: Detects and localizes clothing items in images- **Multi-Category Support**: Handles diverse clothing categories including:    </center>

- **Output**: Bounding boxes with category classifications

  - Coats & Jackets</figure>

### 2. Feature Extraction

- **Custom CNN Architecture**: Multi-layer featurizer model  - Dresses & Skirts

- **Feature Dimension**: 512-dimensional embeddings

- **Framework**: PyTorch with ONNX export for deployment  - Shirts & Tops# Technical Features



### 3. Similarity Matching  - Pants & Shorts* <b>Object Detection Model:</b> Leveraged the power of the YOLOv5 model trained on fashion images to detect fashion objects in images

- **Index Type**: FAISS FlatL2 Index

- **Search Method**: L2 distance-based nearest neighbor search  - Shoes & Accessories* <b>Feature Extraction:</b> Utilized a Convolutional AutoEncoder implemented with PyTorch to extract latent features from detected fashion objects

- **Dataset**: 50,000+ indexed fashion items

  - And more...* <b>Similarity Search Index: </b> Implemented FAISS library to construct an index, facilitating the search for visually similar outfits based on their distinct attributes

## 🛠️ Technology Stack



- **Deep Learning**: PyTorch, ONNX Runtime

- **Computer Vision**: OpenCV, PIL, Ultralytics YOLOv8## 🏗️ Architecture#### For more information on object detection model and feature extraction process, check out my repositories here:

- **Search & Indexing**: FAISS (Facebook AI Similarity Search)

- **Web Framework**: Streamlit* https://github.com/eyereece/yolo-object-detection-fashion

- **Data Processing**: NumPy, Pandas

- **Visualization**: Matplotlib, Plotly### 1. Object Detection Pipeline* https://github.com/eyereece/visual-search-with-image-embedding



## 📊 Performance Metrics- **Model**: YOLOv8 (ONNX format)



The system has been extensively evaluated with the following results:- **Purpose**: Detects and localizes clothing items in images<br>



- **Mean Average Precision (mAP)**: Competitive performance across all categories- **Output**: Bounding boxes with category classifications

- **Retrieval Speed**: Real-time inference (<100ms per query)

- **Scalability**: Handles 50K+ indexed items efficiently# Project Demo

- **Category Accuracy**: High precision in multi-class detection

### 2. Feature Extraction

## 🚀 Getting Started

- **Custom CNN Architecture**: Multi-layer featurizer model#### Online Streamlit Demo:

### Prerequisites

- **Feature Dimension**: 512-dimensional embeddingsTry the [online streamlit demo](https://smartstylist.streamlit.app).

```bash

Python 3.8+- **Framework**: PyTorch with ONNX export for deployment

pip

```<b>Homepage:</b>



### Installation### 3. Similarity Matching



1. **Clone the repository**- **Index Type**: FAISS FlatL2 Index<figure>

```bash

git clone https://github.com/dhriti-kourla/Fashion-Recommendation-System-Using-Deep-Learning-and-Computer-Vision.git- **Search Method**: L2 distance-based nearest neighbor search    <center>

cd Fashion-Recommendation-System-Using-Deep-Learning-and-Computer-Vision

```- **Dataset**: 50,000+ indexed fashion items        <img src="https://static.wixstatic.com/media/81114d_e21c115d1ce141388a4ffc3ecd31c8ad~mv2.gif" alt="preview">



2. **Install dependencies**    </center>

```bash

pip install -r requirements.txt## 🛠️ Technology Stack</figure>

```



3. **Download pre-trained models**

- Place YOLOv8 model (`best.onnx`) in `models/` directory- **Deep Learning**: PyTorch, ONNX Runtime<br>

- Ensure featurizer model (`featurizer-model-1.pt`) is in root directory

- Add FAISS index (`flatIndex.index`) to root directory- **Computer Vision**: OpenCV, PIL, Ultralytics YOLOv8



### Running the Application- **Search & Indexing**: FAISS (Facebook AI Similarity Search)<b>Gallery:</b>



```bash- **Web Framework**: Streamlit

streamlit run home.py

```- **Data Processing**: NumPy, Pandas<figure>



The application will launch in your default browser at `http://localhost:8501`- **Visualization**: Matplotlib, Plotly    <center>



## 📁 Project Structure        <img src="https://static.wixstatic.com/media/81114d_47ce716d2b794785bb3b1b467b2ad425~mv2.gif" alt="preview">



```## 📊 Performance Metrics    </center>

├── home.py                      # Main Streamlit application

├── featurizer_model.py          # Custom CNN feature extraction model</figure>

├── obj_detection.py             # YOLOv8 object detection pipeline

├── test_recommender.py          # Testing and evaluation scriptsThe system has been extensively evaluated with the following results:

├── quick_train.py               # Model training utilities

├── evaluation_metrics.py        # Performance evaluation tools<br>

├── models/

│   ├── best.onnx               # YOLOv8 detection model- **Mean Average Precision (mAP)**: Competitive performance across all categories

│   └── data.yaml               # Model configuration

├── pages/- **Retrieval Speed**: Real-time inference (<100ms per query)<b>Object Detection Model: </b>

│   ├── gallery.py              # Sample results gallery

│   └── TechnicalFeatures.py    # Technical documentation- **Scalability**: Handles 50K+ indexed items efficiently

├── src/

│   ├── featurizer_model.py     # Core feature extraction- **Category Accuracy**: High precision in multi-class detection<figure>

│   └── utilities.py            # Helper functions

├── index_images/               # Indexed fashion dataset    <center>

├── gallery/                    # Sample queries and results

└── requirements.txt            # Python dependencies## 🚀 Getting Started        <img src="https://static.wixstatic.com/media/81114d_f36652e9b7e844869ebb086e5f790beb~mv2.gif" alt="preview" height="500" width="500">

```

    </center>

## 💡 How It Works

### Prerequisites</figure>

1. **Image Upload**: User uploads a fashion image through the web interface

2. **Object Detection**: YOLOv8 detects and crops clothing items

3. **Feature Extraction**: CNN extracts 512-dimensional feature vectors

4. **Similarity Search**: FAISS finds top-k most similar items```bash<br>

5. **Results Display**: System presents visually similar recommendations

Python 3.8+

## 🎨 Use Cases

pip# Getting Started

- **E-commerce**: Product recommendation for online shopping platforms

- **Fashion Discovery**: Help users find similar styles and alternatives```

- **Wardrobe Management**: Organize and match clothing items

- **Style Inspiration**: Discover new fashion combinationsClone the repository: 

- **Visual Search**: Find products based on images rather than text

### Installation```bash

## 📈 Model Training

git clone https://github.com/eyereece/fashion-recommender-cv.git

The system includes training scripts for custom datasets:

1. **Clone the repository**```

```bash

python quick_train.py```bash

```

git clone https://github.com/dhriti-kourla/Fashion-Recommendation-System-Using-Deep-Learning-and-Computer-Vision.gitNavigate to the project directory:

Evaluation metrics can be generated using:

cd Fashion-Recommendation-System-Using-Deep-Learning-and-Computer-Vision```bash

```bash

python evaluation_metrics.py```cd fashion-recommender-cv

```

```

## 🔬 Technical Highlights

2. **Install dependencies**

- **Transfer Learning**: Leverages pre-trained weights for improved performance

- **Efficient Indexing**: FAISS enables sub-linear search complexity```bashInstall dependencies:

- **Production-Ready**: ONNX format ensures cross-platform deployment

- **Modular Design**: Easy to extend with new categories or modelspip install -r requirements.txt```bash

- **Comprehensive Evaluation**: Multiple metrics for performance assessment

```pip install -r requirements.txt

## 📝 Acknowledgments

```

This project was developed as an exploration of deep learning applications in fashion technology, drawing inspiration from recent advances in computer vision and recommendation systems.

3. **Download pre-trained models**

## 📄 License

- Place YOLOv8 model (`best.onnx`) in `models/` directoryRun the streamlit app:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- Ensure featurizer model (`featurizer-model-1.pt`) is in root directory```bash

## 🤝 Contributing

- Add FAISS index (`flatIndex.index`) to root directorystreamlit run home.py

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

```

## 📧 Contact

### Running the Application

**Dhriti Kourla**

- GitHub: [@dhriti-kourla](https://github.com/dhriti-kourla)<br>



---```bash



⭐ If you find this project useful, please consider giving it a star!streamlit run home.py# Usage


```* Upload an image of an outfit (background in white works best)

* It currently only accepts jpg and png file

The application will launch in your default browser at `http://localhost:8501`* Click "Show Recommendations" button to retrieve recommendations

* To update results, simply click on the "Show Recommendations" button again

## 📁 Project Structure* Navigate over to the sidebar, at the "gallery", to explore sample results



```<br>

├── home.py                      # Main Streamlit application

├── featurizer_model.py          # Custom CNN feature extraction model# Dataset

├── obj_detection.py             # YOLOv8 object detection pipeline

├── test_recommender.py          # Testing and evaluation scripts#### The dataset used in this project is available <a href="https://github.com/eileenforwhat/complete-the-look-dataset/tree/master">here</a>:

├── quick_train.py               # Model training utilities<div class="box">

├── evaluation_metrics.py        # Performance evaluation tools  <pre>

├── models/    @online{Eileen2020,

│   ├── best.onnx               # YOLOv8 detection model  author       = {Eileen Li, Eric Kim, Andrew Zhai, Josh Beal, Kunlong Gu},

│   └── data.yaml               # Model configuration  title        = {Bootstrapping Complete The Look at Pinterest},

├── pages/  year         = {2020}

│   ├── gallery.py              # Sample results gallery}

│   └── TechnicalFeatures.py    # Technical documentation  </pre>

├── src/</div>
│   ├── featurizer_model.py     # Core feature extraction
│   └── utilities.py            # Helper functions
├── index_images/               # Indexed fashion dataset
├── gallery/                    # Sample queries and results
└── requirements.txt            # Python dependencies
```

## 💡 How It Works

1. **Image Upload**: User uploads a fashion image through the web interface
2. **Object Detection**: YOLOv8 detects and crops clothing items
3. **Feature Extraction**: CNN extracts 512-dimensional feature vectors
4. **Similarity Search**: FAISS finds top-k most similar items
5. **Results Display**: System presents visually similar recommendations

## 🎨 Use Cases

- **E-commerce**: Product recommendation for online shopping platforms
- **Fashion Discovery**: Help users find similar styles and alternatives
- **Wardrobe Management**: Organize and match clothing items
- **Style Inspiration**: Discover new fashion combinations
- **Visual Search**: Find products based on images rather than text

## 📈 Model Training

The system includes training scripts for custom datasets:

```bash
python quick_train.py
```

Evaluation metrics can be generated using:

```bash
python evaluation_metrics.py
```

## 🔬 Technical Highlights

- **Transfer Learning**: Leverages pre-trained weights for improved performance
- **Efficient Indexing**: FAISS enables sub-linear search complexity
- **Production-Ready**: ONNX format ensures cross-platform deployment
- **Modular Design**: Easy to extend with new categories or models
- **Comprehensive Evaluation**: Multiple metrics for performance assessment

## 📝 Acknowledgments

This project was developed as an exploration of deep learning applications in fashion technology, drawing inspiration from recent advances in computer vision and recommendation systems.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

**Dhriti Kourla**
- GitHub: [@dhriti-kourla](https://github.com/dhriti-kourla)

---

⭐ If you find this project useful, please consider giving it a star!
