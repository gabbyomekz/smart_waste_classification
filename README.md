# 🧠 Smart Waste Classification using CNN (TensorFlow/Keras)

This project applies **Deep Learning and Computer Vision** to classify waste images into **Organic** and **Recyclable** categories. It aims to contribute to **AI-driven sustainability** by supporting smarter recycling systems and automated waste management solutions.

---

## 🗂️  Project Structure
smart_waste_classification/
├── models/             # The best model saved from training
├── notebooks/          # Jupyter notebooks for analysis and modeling
├── outputs/            # Generated outputs like confusion matrix, accuracy and loss plots, model predictions
├── README.md           # Project overview and documentation
├── requirements.txt    # List of Python dependencies for easy setup

## 🗂️ Dataset

**Source:** [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data?resource=download)

* **Total Images:** 25,077

  * 'Train Images': 22,564
  * 'Test Images': 2,513

* **Categories:**

  * `Organic`
  * `Recyclable`
* The dataset is already divided into training and testing folders, simplifying model experimentation.

---

## ⚙️ Project Workflow

### 1. Data Setup & Exploration

* Created training and testing image paths.
* Loaded image samples from both categories.
* Visualized example images to ensure data integrity and class balance.

### 2. Data Preprocessing & Augmentation

Used TensorFlow’s **ImageDataGenerator** to:

* Rescale pixel values (normalization) for train, validation and test sets
* Apply random transformations: rotation, zoom, horizontal flip on train and validation sets
* For application of ImageDataGenerator on test set, Shuffle must be set to "False"
* Improve model generalization and prevent overfitting.

### 3. Model Architecture

Built a **Convolutional Neural Network (CNN)** using TensorFlow/Keras with the following structure:

* **Conv2D + MaxPooling2D** layers (3 was used) [[Ofor hierarchical feature extraction.
* **Dropout** layers for regularization.
* **Flatten + Dense** layers for classification.
* **Sigmoid output layer** with 1 unit (Organic vs Recyclable).

**Optimizer:** Adam, with default learning rate of 0.001
**Loss Function:** Binary Crossentropy
**Metrics:** Accuracy

Early stopping and Model Checkpoint was implemented to halt training once validation performance stopped improving and to save the best model respectively 

### 4. Model Training

* Model trained on the augmented dataset.
* Validation data (20% if the tain dataset) used to monitor overfitting.
* Epochs tuned experimentally for best trade-off between accuracy and convergence speed.

### 5. Evaluation

After training, the model’s performance was analyzed using:

* **Training and Validation Accuracy and Loss plot**
* **Confusion Matrix**
* **Classification Report** (Precision, Recall, F1-score)
* **Sample Prediction Visualization** — showing model predictions vs true labels.

---

## 📊 Results

|        Metric       | Value |
| :-----------------: | :---: |
|  Training Accuracy  |  ~87% |
| Validation Accuracy |  ~92% |
|    Test Accuracy    |  ~91% |

The model achieves 86.73% accuracy on unseen test data, demonstrating strong generalization, correctly classifying most Organic and Recyclable samples.

---

## 🚀 How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/gabbyomekz/smart_waste_classification.git
cd smart_waste_classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook waste_classification.ipynb
```

---

## 💡 Future Improvements

* Integrate **Transfer Learning** (e.g., MobileNetV2, EfficientNet).
* Expand to more categories: `glass`, `metal`, `cardboard`, `paper`.
* Deploy a **Streamlit web app** for interactive waste classification.

---

## 🧰 Tech Stack

* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Visualization:** Matplotlib, Seaborn
* **Data Handling:** NumPy, Pandas
* **Image Processing:** OpenCV, Pillow

---

## 🏆 Key Takeaways

* Learned how to build, train, and evaluate CNNs on real-world image data.
* Applied effective data augmentation strategies.
* Demonstrated how deep learning can support environmental sustainability.

---

## 👨‍💻 Author

Developed by [Gabriel Omeke]
📧 Contact: [gabrielomeke92@gmail.com](mailto:gabrielomeke92@gmail.com)
🔗 GitHub: [github.com/gabbyomekz](https://github.com/gabbyomekz)

---

### 🏁 License

This project is released under the **MIT License**. You are free to use, modify, and distribute it with attribution.

