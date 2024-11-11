# RAG
RAG system built on the Milvus vector database.

**This project represents only an introduction to Milvus and the creation of a RAG system. For each individual case, task and data, the process of creating a RAG system can be modified.**

 - Milvus startup and configuration procedure is described in the official documentation https://milvus.io/docs/install_standalone-docker.md. 

 - Google's Gemini model was used as the GPT model https://gemini.google.com/app.

 - All necessary APIs need to be put in your .env file.

 - All required dependencies and their versions can be found in the requirements.txt file.
 - Python 3.12.3 was used for this project.

 - As an example PDF file, I used How to Write a Better Thesis from https://www.pdfdrive.com/how-to-write-a-better-thesis-e25132067.html. 

 - It is worth noting that the function of text processing, after retrieving from a PDF file, each may be different and different from the one presented.

 - The main notebook contains the complete model development cycle with explanations
 - It describes the process of building a Milvus collection, receiving a response from the RAG system, and visualizing the RAG system's decision-making process

Below is an example of RAG system Embeddings in 2d format.

![image](https://github.com/user-attachments/assets/e99d9eec-ecc2-433d-ac02-35bf5d86f9c0)

There is also a separate make_collection file, which is needed to create a Milvus collection separately.

An app file has been created that creates a web page on your locahost. This page mimics how the RAG system works in production.

Below is a screenshot of an example RAG system question and answer. The wait for a response can be a few seconds.

![image](https://github.com/user-attachments/assets/814a10d2-1773-494a-a9e0-93615370576f)

Thank you for your attention!
