# From-Scratch Language Model

A minimal next-word LSTM trained on a domain-specific corpus of cooking recipes.  
Includes raw-text preprocessing, vocabulary encoding, sliding-window sequence generation, one-hot encoding, model training, and text generation.

**Tools and Frameworks:**  
- Python  
- NumPy  
- Keras (TensorFlow backend)  

---

## 📂 Dataset and Preprocessing  
We use a recipe dataset containing **4,869 recipes** stored in a single text file.  

**Preprocessing workflow:**  
1. Process the dataset to extract all unique words and build a vocabulary.  
2. One-hot encode all words using the vocabulary.  
3. Segment the dataset into 50-word one-hot–encoded sequences (treated as “sentences”).  

---

## 🧠 Model Training  
The neural network is trained using:  
- **Input:** 50-word encoded sequences.  
- **Target:** The next word in the dataset.  

Training was performed on a GPU and required **over 12 hours** to reach the current results.

---

## ✍️ Text Generation  
Once trained, the model can generate new recipes by:  
1. Taking a randomly selected 50-word sentence as a seed.  
2. Iteratively predicting and appending the next word until a complete recipe is formed.  

---

## ✅ Results  

### Strengths  
- Generates an appropriate recipe title.  
- Correctly formats the amount/measure/ingredients table.  
- Assigns sensible quantities to ingredients.  
- Produces generally coherent instructions.  

### Limitations  
- The generated title is often unrelated to the recipe content. 
- Occasionally references ingredients in the instructions that are not listed in the ingredients table.  

---

## 🚀 Future Improvements  
- Further training to improve consistency between ingredient lists and instructions.  
- Expanding the dataset for richer vocabulary and recipe diversity.  

---
