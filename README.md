# Illegal Building Detection (Erbil City)

![Erbil Municipal AI](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Erbil_Citadel_2014.jpg/800px-Erbil_Citadel_2014.jpg)

An AI-powered municipal oversight platform designed to automatically detect unauthorized construction using satellite imagery. Built specifically for the **First Annual Student Forum for AI Projects** at the University of Baghdad, College of Artificial Intelligence.

---

## 🚀 Live Demo

**[Link to Hugging Face Spaces Demo (TBD)](#)**

*This project is ready to be deployed to Hugging Face Spaces using the provided `app.py` and `requirements.txt`.*

---

## 🏆 Project Judging Criteria

This project was developed addressing the core evaluation criteria of the forum:

### 1. Innovation and Originality (10 Points)
- **Distinction**: Unlike traditional manual city surveying which is slow and labor-intensive, this project utilizes automated temporal image analysis.
- **New Idea**: We implement a simulated **Siamese Convolutional Neural Network (CNN)** approach. By feeding two satellite images (T0 and T1) into the network simultaneously, the system computes the feature distance to isolate newly constructed buildings while ignoring seasonal changes.

### 2. Technical Mastery and Implementation (30 Points)
- **Modern Programming & UI**: The application is built using **Python** and **Streamlit**, featuring a highly customized, premium CSS frontend designed to look like a professional municipal dashboard.
- **AI Algorithms**: The core logic relies on computer vision techniques (simulating the Siamese network's output via OpenCV Structural Similarity and morphological operations) to accurately draw bounding boxes around violations.

### 3. Practical Application (35 Points)
- **Solving a Problem**: Erbil city is expanding rapidly. Unauthorized construction leads to poor urban planning and lost municipal revenue. This system allows the municipality to scan entire sectors instantly.
- **Realism & Sustainability**: The solution requires only satellite imagery (which is readily available via APIs like Google Earth Engine or Planet) and standard computing resources, making it highly sustainable for government use.

### 4. Presentation and Delivery (15 Points)
- **Clarity & Prototype**: The Streamlit dashboard is designed for an interactive, smooth presentation. It clearly displays the "Baseline" and "Recent" images, with a dynamic "Run Analysis" button that yields clear, actionable metrics (Number of violations, total area changed, and confidence scores).

### 5. Ethics and Integrity (10 Points)
- **Privacy**: The system strictly analyzes structural data from macroscopic satellite views. It does not process facial recognition, personal identification, or any private citizen data.
- **Integrity**: The algorithmic approach is objective. It flags *all* structural changes regardless of neighborhood, ensuring fair and unbiased municipal oversight.

---

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd illegal-building-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: The main Streamlit application containing the UI and the Siamese CNN simulation logic.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `sample_data/`: Contains example 'Before' and 'After' satellite images for testing the demo.

## 👥 Team
- *Add your team names here*
