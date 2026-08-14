# **AI Engineer & Data Scientist**

### Mohamed Atef

📍 Qesm Damanhour - El Beheira, Egypt | 📞 +201027139937 | 📧 [e.muhammadatef@gmail.com](https://www.google.com/search?q=mailto%3Ae.muhammadatef%40gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/engmuhammadatef) | [GitHub](https://github.com/engmuhammadatef) | [Kaggle](https://www.kaggle.com/muhammadatef)

---

AI Engineer with 2 years of hands-on experience building and deploying production-grade AI systems, with a strong focus on Large Language Models (LLMs) and NLP. Practically skilled across Retrieval-Augmented Generation (RAG), AI Agents, fine-tuning, and training LLMs from scratch, and experienced integrating embeddings, vector stores and tokenization into scalable retrieval and inference pipelines. Complementary experience in computer vision and recommendation systems, plus solid ML fundamentals and model-evaluation practices. Strong software-engineering approach to delivery — building APIs and services (Flask; familiar with FastAPI), containerized deployments (Docker), cloud environments (Azure; familiar with AWS), and LLMOps for monitoring, evaluation and continuous improvement. Comfortable working cross-functionally to turn user pain points into reliable, cost-aware AI features used in production.

---

# **Work Experience**

### AI Engineer

**ComplyMarket** | **Alexandria, Hybrid**


**November 2024 – Present**

* **Technical Lead & Core Developer for Abkarino Suite:** Engineered ComplyMarket’s proprietary domain-specific AI models for chemistry and scientific content from scratch in PyTorch.


    1- **Abkarino LLM (Decoder):** Designed custom Transformer architectures, automated dataset collection/scraping, designed custom tokenizers, and managed distributed multi-GPU pre-training (FSDP), SFT, and alignment (GRPO) for high-factuality generation.


    2- **Abkarino Embedding (Encoder):** Developed a specialized domain encoder model using contrastive representation learning to generate dense vector embeddings for scientific literature, specialized terminology, and chemical notations, optimizing downstream domain retrieval.


* **ComplyMarket Chatbot & Multi-Cloud RAG System:** Designed and shipped the enterprise Chatbot RAG pipeline originally on Azure, then spearheaded a seamless multi-cloud migration to Google Cloud Platform (GCP) to optimize multi-tenant retrieval performance and infrastructure costs.


* **Event-Driven & Scheduled ETL Pipelines:** Implemented automated vector storage synchronization using GCP Cloud Functions (EventArc) to trigger real-time vector indexing/deletion on GCS blob events, paired with 6-hour Cloud Scheduler HTTP reconciliation functions to keep Vertex AI Vector Search in sync with raw storage.


* **MLOps & Technical Leadership:** Owned containerized microservice deployments (Docker/Flask), continuous LLM evaluation loops, and reproducible experiment practices.


* **Cross-Functional Alignment:** Aligned cross-functional teams (Engineering, Product, Design) and mentored junior engineers to drive user-focused delivery.



### AI / Machine Learning Engineer



**Cyber Royale** | **UAE, Remote - Contract**


**May 2024 – July 2024**

* **Content Moderation Pipeline:** Built and shipped a production-oriented content moderation system: collected and preprocessed structured/unstructured data for real-time social media inference.


* **NLP Moderation Model:** Developed a high-accuracy ML moderation model (Bag-of-Words feature engineering + SVC) achieving 97.5% accuracy for offensive language and hate-speech detection while mitigating class imbalance.


* **Computer Vision & Multilingual OCR:** Implemented CNN moderation architectures achieving 95% accuracy for violence/nudity detection; integrated OCR and translation APIs for multi-language text-on-image moderation.

---

# **Education**

### Bachelor’s degree in Computer Science | Damanhur University

*October 2020 - June 2024*

* **Relevant Coursework**: Software Engineering, Data Structures & Algorithms, MATH 3, Statistics & Probability, Advanced Databases, Data Mining, Artificial Intelligence
* **Activities**: Member at Google Developer Student Clubs (GDSC)
* **Competitive Programming Mentor**: Led workshops, trained 30–50 students in problem solving, organized Codeforces competitions, and awarded achievements.

---

# **Projects**

## Abkarino.com — Domain LLM (Decoder) & Embedding Model (Encoder) (Built from Scratch)



* Technical Lead & Core Developer of a dual-model domain AI framework in PyTorch: built a generative Decoder LLM for complex scientific reasoning and an Encoder Embedding Model for dense vector representation and semantic retrieval.


* Developed high-throughput scraping/parsing scripts and designed a specialized tokenizer tailored to scientific vocabulary and SMILES/chemical notations.


* Managed multi-node distributed pre-training (FSDP), post-training alignment (SFT & GRPO), and evaluation benchmarks to maximize domain factuality.



## ComplyMarket Chatbot & RAG Infrastructure (Migrated Azure → GCP)



* Engineered an enterprise multi-tenant RAG chatbot system, leading the architectural migration from Microsoft Azure to GCP (Vertex AI Vector Search, Google Cloud Storage, Cloud Functions).


* Built scalable Flask streaming endpoints supporting dynamic prompt reconstruction, real-time file upload parsing, dynamic model selection, and multi-threaded heartbeat status responses.


* Designed event-driven EventArc triggers and automated 6-hour GCS reconciliation jobs to maintain vector index consistency; upgraded baseline search into an agentic pipeline with web search and pre-knowledge retrieval.



## Content Moderation System — NLP & Computer Vision



* Implemented text moderation using Bag-of-Words + SVC achieving 97.5% accuracy on offensive-language detection.


* Built image/video moderation pipelines leveraging CNNs and MobileNetV2 variants achieving 95% accuracy for violence detection, integrated with OCR and translation engines.



## Study Partner — Recommendation System & ETL Optimization



* Designed a hybrid recommender combining content-based and collaborative filtering to match users with study partners and update matches in near-real time.


* Implemented an ETL migration from MongoDB → PostgreSQL, reducing data retrieval/transformation latency by ~87%, significantly improving responsiveness.


* Built automated preference-updating and partner-matching pipelines to maintain recommendation relevance.



## Chief Financial Officer AI System (Generative AI)

* Built an LLM- and RAG-powered assistant for CFO workflows with dynamic visualizations, accurate time-series queries (e.g., revenue/expenses charts & tables), and smart metric suggestions for decision support.

## Text Classification (From BagOfWords to Transformers)

[GitHub LINK](https://github.com/EngMuhammadAtef/Text-Classification-From-BagOfWords-To-Transformers)

* Built toxicity detection pipeline: BoW + SVM, RNNs, Bi-LSTM, GRU, and Transformers; benchmarked models for offensive comment classification.

## Video Classification Model

* Fine-tuned EfficientNetB0 on UCF101 (92% accuracy). Built OpenCV + TensorFlow preprocessing and deployed real-time inference for moderation.

## House Prices Advanced Regression

[GitHub LINK](https://github.com/EngMuhammadAtef/House-Pricing)

* Regression models with PCA, feature engineering, XGBoost and Power BI (91.75% accuracy).

---

# **Skills & Tools**

### Programming & Software Engineering



* Python, OOP, Data Structures & Algorithms, API Development (Flask, familiar with FastAPI), Multi-threading, Distributed Systems, Git, Docker, CI/CD.



### Large Language Models & NLP



* LLM Pre-training from Scratch, Custom Tokenizer Design, PyTorch Transformer Architectures, Distributed Multi-GPU Training (FSDP), Alignment (SFT, GRPO), AI Agents, RAG, LangChain, Hugging Face, Model Evaluation & Checkpointing.



### Cloud Architecture & Infrastructure



* Multi-cloud deployment across Azure (Functions, AI Search, App Services) and GCP (Vertex AI Vector Search, Cloud Functions, GCS, EventArc, Cloud Scheduler), featuring Docker orchestration, multi-tenant data isolation, production-grade reliability, and full-stack integrations.



### Machine Learning & LLMOps



* Monitoring & Evaluation Pipelines, Containerized Inference Deployments, Feedback Loops, Experiment Reproducibility, Classical ML, CNNs, Recommendation Systems.



### Data Engineering & Processing



* Automated Web Scraping, Structured/Unstructured ETL, PostgreSQL, MySQL, MongoDB, Vector Databases, LLM Embeddings.



### Mathematics for AI



* Statistics, probability, linear algebra, calculus.



### Soft Skills



* Cross-functional Leadership, Technical Project Management, Communication, Problem Solving.



---

# **Certifications**

* 📜 IBM Data Science Professional Certificate
[View Certificate](https://www.coursera.org/account/accomplishments/professional-cert/CXXTC39BSENB)
* 📜 DataCamp Associate Data Scientist Certificate
[View Certificate](https://www.datacamp.com/certificate/DSA0017479995176)
* 📜 Stanford Machine Learning Certificates
[Supervised ML](https://www.coursera.org/account/accomplishments/verify/MGAEGVC53FGC) | [Advanced Algorithms](https://www.coursera.org/account/accomplishments/verify/SV3DFEWKQN8C)
* 📜 ITI NLP Engineer Courses
[View Certificate](https://maharatech.gov.eg/mod/customcert/view.php?id=13142&downloadown=1)

---

# **Languages**

* **Arabic**: Native
* **English**: Professional Working Proficiency
