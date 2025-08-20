# GHGNN-Master

The code and weights have been released, enjoy it！ You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the model prediction and real-time performance, **run:  test.py!**

**Run test.py to view the model's average displacement error (ADE), final displacement error (FDE), model parameters, GPU memory allocated, GPU memory reserved, Average inference time per run, and FPS.**

It is worth noting that GPU memory allocation, GPU memory reserved, Average inference time per run, and FPS may vary depending on the experimental equipment.

If you can't run test.py correctly, please contact me in time!

## Environment
The code is trained and tested on RTX 4090, Python 3.8.13, numpy 1.23.1, pytorch 2.3.0 and CUDA.

## Code Structure
checkpoint folder: contains the trained model weights

dataset folder: contains ETH, UCY and SDD datasets

model.py: the code of GHGNN

test.py: for testing the code

utils.py: general utils used by the code

metrics.py: Measuring tools used by the code

## Model Evaluation
You can easily run the model！ To use the pretrained models at checkpoint/ and evaluate the models performance run:  test.py

## Acknowledgement
Some codes are borrowed from Social-STGCNN, SGCN, IMGCN and DSTIGCN. We gratefully acknowledge the authors for posting their code.
