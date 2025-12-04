# Fashion Recommendation System Using Deep Learning and Computer Vision<h2 align="center">SmartStylist: A Fashion Recommender System powered by Computer Vision</h2>

<br>

An intelligent fashion recommendation system that leverages state-of-the-art deep learning techniques and computer vision to analyze clothing images and provide personalized style suggestions based on visual similarity.

<center>

## 🎯 Project Overview<a href="https://www.joankusuma.com/post/smart-stylist-a-fashion-recommender-system-powered-by-computer-vision"><img src='https://img.shields.io/badge/Project_Page-SmartStylist-pink' alt='Project Page'></a> 

<a href='https://www.joankusuma.com/post/object-detection-model-yolov5-on-fashion-images'><img src='https://img.shields.io/badge/Project_Page-ObjectDetection-blue' alt='Object Detection'></a> 

This system combines object detection, deep learning-based feature extraction, and efficient similarity search to deliver accurate fashion recommendations. The model analyzes uploaded fashion images, detects clothing items, extracts visual features, and retrieves the most similar items from a curated dataset.<a href='https://www.joankusuma.com/post/powering-visual-search-with-image-embedding'><img src='https://img.shields.io/badge/Project_Page-VisualSearch-green'></a> 

<a href='https://smartstylist.streamlit.app'><img src='https://img.shields.io/badge/Streamlit-Demo-red'></a>

## ✨ Key Features</center>

<br>

- **Advanced Object Detection**: Utilizes YOLOv8 for precise clothing item detection and classification<br>

- **Deep Feature Extraction**: Custom CNN-based featurizer model for robust visual embeddings<figure>

- **Efficient Similarity Search**: FAISS-powered vector indexing for real-time recommendations    <center>

- **Interactive Web Interface**: Streamlit-based UI for seamless user experience        <img src="https://static.wixstatic.com/media/81114d_7f499b8207b848bc8bccfe1035a28b3d~mv2.png" alt="flowchart" height="350" width="600">

- **Multi-Category Support**: Handles diverse clothing categories including:    </center>

  - Coats & Jackets</figure>

  - Dresses & Skirts

  - Shirts & Tops# Technical Features

  - Pants & Shorts* <b>Object Detection Model:</b> Leveraged the power of the YOLOv5 model trained on fashion images to detect fashion objects in images

  - Shoes & Accessories* <b>Feature Extraction:</b> Utilized a Convolutional AutoEncoder implemented with PyTorch to extract latent features from detected fashion objects

  - And more...* <b>Similarity Search Index: </b> Implemented FAISS library to construct an index, facilitating the search for visually similar outfits based on their distinct attributes



## 🏗️ Architecture#### For more information on object detection model and feature extraction process, check out my repositories here:

* https://github.com/eyereece/yolo-object-detection-fashion

### 1. Object Detection Pipeline* https://github.com/eyereece/visual-search-with-image-embedding

- **Model**: YOLOv8 (ONNX format)

- **Purpose**: Detects and localizes clothing items in images<br>

- **Output**: Bounding boxes with category classifications

# Project Demo

### 2. Feature Extraction

- **Custom CNN Architecture**: Multi-layer featurizer model#### Online Streamlit Demo:

- **Feature Dimension**: 512-dimensional embeddingsTry the [online streamlit demo](https://smartstylist.streamlit.app).

- **Framework**: PyTorch with ONNX export for deployment

<b>Homepage:</b>

### 3. Similarity Matching

- **Index Type**: FAISS FlatL2 Index<figure>

- **Search Method**: L2 distance-based nearest neighbor search    <center>

- **Dataset**: 50,000+ indexed fashion items        <img src="https://static.wixstatic.com/media/81114d_e21c115d1ce141388a4ffc3ecd31c8ad~mv2.gif" alt="preview">

    </center>

## 🛠️ Technology Stack</figure>



- **Deep Learning**: PyTorch, ONNX Runtime<br>

- **Computer Vision**: OpenCV, PIL, Ultralytics YOLOv8

- **Search & Indexing**: FAISS (Facebook AI Similarity Search)<b>Gallery:</b>

- **Web Framework**: Streamlit

- **Data Processing**: NumPy, Pandas<figure>

- **Visualization**: Matplotlib, Plotly    <center>

        <img src="https://static.wixstatic.com/media/81114d_47ce716d2b794785bb3b1b467b2ad425~mv2.gif" alt="preview">

## 📊 Performance Metrics    </center>

</figure>

The system has been extensively evaluated with the following results:

<br>

- **Mean Average Precision (mAP)**: Competitive performance across all categories

- **Retrieval Speed**: Real-time inference (<100ms per query)<b>Object Detection Model: </b>

- **Scalability**: Handles 50K+ indexed items efficiently

- **Category Accuracy**: High precision in multi-class detection<figure>

    <center>

## 🚀 Getting Started        <img src="https://static.wixstatic.com/media/81114d_f36652e9b7e844869ebb086e5f790beb~mv2.gif" alt="preview" height="500" width="500">

    </center>

### Prerequisites</figure>



```bash<br>

Python 3.8+

pip# Getting Started

```

Clone the repository: 

### Installation```bash

git clone https://github.com/eyereece/fashion-recommender-cv.git

1. **Clone the repository**```

```bash

git clone https://github.com/dhriti-kourla/Fashion-Recommendation-System-Using-Deep-Learning-and-Computer-Vision.gitNavigate to the project directory:

cd Fashion-Recommendation-System-Using-Deep-Learning-and-Computer-Vision```bash

```cd fashion-recommender-cv

```

2. **Install dependencies**

```bashInstall dependencies:

pip install -r requirements.txt```bash

```pip install -r requirements.txt

```

3. **Download pre-trained models**

- Place YOLOv8 model (`best.onnx`) in `models/` directoryRun the streamlit app:

- Ensure featurizer model (`featurizer-model-1.pt`) is in root directory```bash

- Add FAISS index (`flatIndex.index`) to root directorystreamlit run home.py

```

### Running the Application

<br>

```bash

streamlit run home.py# Usage

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
