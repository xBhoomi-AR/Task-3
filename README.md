# Attention is All You Need

## Methodology

In this task we implement a Transformer-based Neural Machine Translation model using PyTorch. The model follows an encoder-decoder architecture consisting of token embeddings, positional encoding, multi-head self-attention, feed-forward networks, residual connections, and layer normalization.

The model was trained on the Multi30k English-German dataset containing approximately:

* Training Sentences: 29,000 approx
* Validation Sentences: 1,014 approx
* Test Sentences: 1,000 approx

Training was performed using:

* Optimizer: Adam
* Loss Function: Cross Entropy Loss
* Batch Size: 32
* Epochs: 10

I used Beam Search decoding to generate translations. Model performance was then evaluated using Validation Loss and BLEU Score.

---

## Dependencies

We require pytorch, numpy, math and nltk libraries

---

## How to Run

### Training the Model

We have to run :
 !python train.py for (google colab)

This trains the Transformer model and saves the trained weights as:
 transformer.pt

### To Validate the Model

We have to run:
!python val.py

This evaluates the model on the validation dataset and reports the validation loss.

### Testing the Model

We have to run:
!python test.py

This generates sample translations and computes the BLEU score on the test dataset.

---

## What to Expect

### Training Output


Epoch 1 Loss: 4.70
...
Epoch 10 Loss: 0.29
Model saved.


### Validation Output


Validation Loss: 0.29

### Testing Output


 SAMPLE TRANSLATIONS :

Sample 1
Prediction: ein mann spielt fußball

Average BLEU (0–100): 25.71


### Final Results

* Final Training Loss: 0.295
* Validation Loss: 0.293
* BLEU Score: 25.71

### Output SS
<img width="816" height="377" alt="WhatsApp Image 2026-06-25 at 10 50 29 PM" src="https://github.com/user-attachments/assets/64050fe7-d0f7-4332-8567-b052f433398e" />

<img width="805" height="442" alt="WhatsApp Image 2026-06-25 at 10 50 01 PM" src="https://github.com/user-attachments/assets/92d1318a-30ec-4c8a-891e-45e9aadc187d" />

<img width="812" height="690" alt="WhatsApp Image 2026-06-25 at 10 51 13 PM" src="https://github.com/user-attachments/assets/e646e299-b55a-4f14-91fd-0476f9017930" />

