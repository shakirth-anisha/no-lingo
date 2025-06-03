<p align="center">
  <h1 align="center">NO-LINGO</h1>
  <p align="center"><strong>Empowering communication through seamless gesture recognition.</strong></p>

  <p align="center">
    <img src="https://img.shields.io/github/last-commit/shakirth-anisha/no-lingo" alt="Last Commit" />
    <img src="https://img.shields.io/github/languages/top/shakirth-anisha/no-lingo" alt="Top Language" />
    <img src="https://img.shields.io/github/languages/count/shakirth-anisha/no-lingo" alt="Language Count" />
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge" />
  </p>
</p>

---


## Table of Contents

- [Overview](#overview)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Testing](#testing)  

---

## Overview

**no-lingo** is a cutting-edge developer tool that empowers users to harness gesture recognition for real-time communication and image classification.

### Why NO-LINGO?

This project aims to simplify the integration of gesture-based interactions into applications. The core features include:

- ✋ **Gesture Recognition**: Processes image contours to translate gestures into text, enhancing accessibility.
- 📊 **SVM Model Training**: Leverages Histogram of Oriented Gradients (HOG) for effective image classification.
- 🎥 **Real-time Sign Language Recognition**: Uses live video to detect and translate hand gestures instantly.
- 📸 **Image Dataset Creation**: Captures and processes images to build robust, custom datasets.
- 🖌️ **Skin Detection**: Improves recognition by isolating hand regions using skin tone detection.

---

## Getting Started

### Prerequisites

Ensure the following are installed:

- **Python** (recommended: 3.8+)
- **Conda** (for environment management)

---

## Installation

Build **no-lingo** from source and install dependencies.

1. Clone the repository:

   ```bash
   git clone https://github.com/shakirth-anisha/no-lingo
   ```
2. Navigate to the project directory:

   ```bash
   cd no-lingo
   ```
3. Create and activate the environment:

  ```bash
  conda env create -f conda.yml 
  conda activate no-lingo
  ```

---

## Usage

To run the application:

  ```bash
  conda activate no-lingo
  python ASL.py
  ```

---

⬆ [Back to Top](#no-lingo)
