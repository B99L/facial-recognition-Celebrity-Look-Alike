[16.05 16:08] Roccia Nevio (roccinev)

# Celebrity Look-Alike Finder

This project is a big data application that uses facial recognition technology to find a user's celebrity look-alike. It takes a user's picture as input and compares it to a dataset of celebrity images. The application is designed to be both entertaining and a showcase of the power of distributed computing and facial recognition technology.

 

## Features

- Facial feature extraction using DeepFace

- Similarity computation using FAISS and Python

- Performance comparison between distributed (Spark) and sequential (single machine) processing

 

## Dataset

We use a subset of the CelebA dataset, a large-scale celebrity dataset with over 200,000 celebrity images. Each image is annotated with 40 attribute labels.

 
## Requirements

- Python 3.7 or later

- Apache Spark

- DeepFace

- FAISS


## Usage

- For partial processing of feature extraction, open the file `Parallel Processing.ipynb`.

- For sequential feature extraction, open `Sequential Processing.ipynb`.

- For similarity calculation with FAISS and iteration, open `Faiss&Brute-Force-Iteration.ipynb`.

- For the website application, open `app.py` and in the templates folder open `index.html`.
 

## Limitations

Due to the limitations of the Spark cluster, we were only able to use 10,000 images out of the more than 200,000 images in the CelebA dataset.

 
## Conclusion and Future Work

We hope you find this Celebrity Look-Alike Finder project interesting and useful. It's a fantastic way to understand and utilize cutting-edge technologies such as DeepFace, FAISS, and Apache Spark in the world of big data. 

Despite its current limitations, we believe this project has substantial potential for future development and improvement. Expanding the current dataset beyond the 10,000 images we've used, implementing more efficient feature extraction methods, and improving the user interface are just a few of the enhancements we are considering. 

We look forward to refining and expanding this project, and we warmly welcome any feedback or contributions you might have to make. We believe that collaboration and open-source development are the keys to driving innovation and improving technology for everyone.

Thank you for your interest in our Celebrity Look-Alike Finder. If you have any questions, suggestions, or concerns, please don't hesitate to reach out to us.

Happy Coding!


 








