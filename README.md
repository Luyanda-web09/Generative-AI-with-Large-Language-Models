# Generative-AI-with-Large-Language-Models

Generative AI using large language models (LLMs) and aims to equip learners with the technical foundations and practical skills necessary for building applications.

Course Overview

Introduction to LLMs: The course highlights the significance of LLMs in developing machine learning and AI applications more efficiently than before.
Project Lifecycle: It covers the generative AI project lifecycle, including model training, instruction tuning, and fine-tuning.
Course Structure

Week 1: Learners will explore the transformer architecture, model training, compute resources, and prompt engineering techniques.
Week 2: Focus on adapting pre-trained models to specific tasks through instruction fine-tuning.
Week 3: Aligning language model outputs with human values and hands-on labs to apply learned techniques.
Hands-on Labs

Lab 1: Compare prompts for dialogue summarization and explore inference parameters.
Lab 2: Fine-tune a large language model from Hugging Face using both full and parameter-efficient fine-tuning.
Lab 3: Implement reinforcement learning from human feedback to classify model responses as toxic or non-toxic.

Week 1

The fundamentals of transformer networks and the Generative AI project lifecycle.

Understanding Transformer Networks

The lecture introduces the transformer architecture, emphasizing its complexity and the significance of the 2017 paper "Attention is All You Need."
Key concepts such as self-attention and multi-headed self-attention are discussed to explain how transformers understand language and process data efficiently.
Generative AI Project Lifecycle

The course outlines the stages of developing Generative AI applications, including decisions on using pre-trained models versus training custom models.
It highlights the importance of selecting the right model size based on specific use cases, noting that smaller models can still achieve impressive results for targeted tasks.

Introducing the fundamentals of generative AI, particularly focusing on large language models (LLMs) and their applications.

Understanding Large Language Models

Large language models are trained on vast datasets, learning to mimic human-like content generation.
These models, with billions of parameters, can perform complex tasks and exhibit emergent properties beyond simple language processing.
Interacting with Language Models

Users interact with LLMs using natural language prompts, which the models process to generate text completions.
The context window for prompts varies by model, allowing for a few thousand words of input.
Project Lifecycle and Applications

The course outlines a project lifecycle for generative AI, including model selection, fine-tuning, and deployment.
Learners will explore how to apply LLMs to solve real-world business and social challenges through practical examples and labs.

The various applications and capabilities of Large Language Models (LLMs) and generative AI.

Applications of LLMs

LLMs can perform a range of tasks beyond chatbots, such as writing essays and summarizing dialogues.
They are also used for translation tasks, converting natural language to machine code, and generating code snippets.
Focused Tasks and Information Retrieval

LLMs can execute specific tasks like named entity recognition, identifying people and places in text.
The models leverage their understanding of language to accurately retrieve and classify information.
Integration with External Data

There is ongoing development in connecting LLMs to external data sources and APIs.
This integration allows models to access updated information and interact with real-world applications.
Model Scaling and Understanding

The performance of LLMs improves as the scale of the models increases, enhancing their language understanding.
Smaller models can also be fine-tuned for specific tasks, making them effective for targeted applications.

The evolution of generative algorithms, particularly focusing on the transition from recurrent neural networks (RNNs) to transformer architecture.

Generative Algorithms and RNNs

RNNs were previously used for generative tasks but had limitations in compute and memory.
RNNs struggled with next-word prediction due to insufficient context from only a few preceding words.
Challenges in Language Understanding

Language complexity includes homonyms and syntactic ambiguity, making predictions difficult.
Context is crucial for understanding meaning, as illustrated by ambiguous sentences.
Introduction of Transformer Architecture

The 2017 paper "Attention is All You Need" introduced the transformer architecture, revolutionizing generative AI.
Transformers can efficiently scale, process data in parallel, and learn the meaning of words, enhancing their predictive capabilities.

The transformer architecture, which significantly enhances the performance of natural language processing tasks.

Transformer Architecture Overview

The transformer architecture improves upon earlier models like RNNs by using self-attention to understand the context and relevance of words in a sentence.
Attention weights are learned during training, allowing the model to determine the importance of each word in relation to others.
Model Components

The architecture consists of two main parts: the encoder and the decoder, which work together to process input data.
Text input must be tokenized into numerical representations before being processed, using methods that can represent whole words or parts of words.
Self-Attention Mechanism

Self-attention allows the model to analyze relationships between tokens, capturing contextual dependencies.
Multiple self-attention heads learn different aspects of language, enhancing the model's understanding of various linguistic features.
This summary encapsulates the key concepts of the transformer architecture and its components, emphasizing the importance of self-attention in natural language processing.

An overview of the transformer architecture, focusing on its prediction process, particularly in the context of a translation task.

Understanding the Prediction Process

The transformer model translates a French phrase into English by tokenizing the input and passing it through the encoder.
The encoder generates a deep representation of the input, which influences the decoder's self-attention mechanisms.
Components of the Transformer Architecture

The architecture consists of an encoder and decoder, where the encoder encodes input sequences and the decoder generates new tokens based on the encoder's context.
Variations of the architecture include encoder-only models (e.g., BERT) for classification tasks and decoder-only models (e.g., GPT) for general tasks.
Applications and Future Learning

The overview emphasizes understanding different transformer models and their applications in real-world scenarios.
The course will further explore prompt engineering, allowing interaction with transformer models through natural language.

Various methods and configuration parameters that influence next-word generation in large language models (LLMs).

Configuration Parameters

These parameters are used during inference to control aspects like the maximum number of tokens generated and the creativity of the output.
The "max new tokens" parameter limits the number of tokens the model generates, but it is not a strict cap.
Decoding Methods

Greedy decoding selects the word with the highest probability, which can lead to repetitive outputs.
Random sampling introduces variability by selecting words based on their probability distribution, reducing repetition but potentially leading to less coherent outputs.
Sampling Techniques

Top k sampling restricts the model to the k most probable tokens, enhancing coherence while allowing some randomness.
Top p sampling limits options based on cumulative probabilities, ensuring the model selects from a sensible range of words.
Temperature Control

The temperature parameter adjusts the randomness of the output: lower values lead to more predictable text, while higher values increase variability and creativity.
A temperature of one uses the default probability distribution, balancing randomness and coherence.
This foundational knowledge prepares you for developing and launching LLM-powered applications in future lessons.

The generative AI project life cycle, specifically for developing and deploying applications powered by large language models (LLMs).

Project Life Cycle Overview

The life cycle includes stages from project conception to launch, emphasizing the importance of defining the project scope accurately.
Understanding the specific tasks the LLM will perform is crucial, as it can save time and compute costs.
Model Development Decisions

Initial decisions involve whether to train a model from scratch or use an existing base model, with a general preference for starting with existing models.
Performance assessment and potential additional training are necessary, including techniques like prompt engineering and fine-tuning.
Deployment and Evaluation

Ensuring the model behaves well and aligns with human preferences is vital, with techniques like reinforcement learning with human feedback introduced.
Evaluation metrics and benchmarks will be explored to assess model performance, and the deployment stage includes optimizing the model for resource efficiency.

The practical aspects of using the lab environment for hands-on learning in the context of generative AI.

Lab Environment Overview

The lab is conducted in Vocareum, which allows access to Amazon SageMaker for running notebooks at no cost.
Students have 2 hours to complete each lab, and they can simply close the browser when finished.
Getting Started with the Lab

The generative AI project life cycle, particularly the selection and training of large language models (LLMs).

Model Selection

You can choose to work with an existing model or train your own from scratch, depending on your use case.
Open-source model hubs, like those from Hugging Face and PyTorch, provide curated models and model cards detailing their use cases and limitations.
Training Process

The initial training phase, known as pre-training, involves learning from vast amounts of unstructured textual data to develop a statistical representation of language.
Different transformer model architectures (encoder-only, decoder-only, and sequence-to-sequence) are trained for specific tasks, such as sentence classification or text generation.
Model Variants

Autoencoding models (e.g., BERT) use masked language modeling for tasks requiring bi-directional context.
Autoregressive models (e.g., GPT) predict the next token based on previous tokens, suitable for text generation.
Sequence-to-sequence models (e.g., T5) utilize both encoder and decoder for tasks like translation and summarization.
Overall, understanding these models and their training objectives will help you select the best model for your generative AI application.
To begin, students need to click "Start Lab" and ensure the AWS status changes to green before proceeding.
The quickest way to access SageMaker is by using the search function in the AWS console.
Using Jupyter Notebooks

Students will work in JupyterLab, where they can run code cells using Shift+Enter or run all cells at once.
It is recommended to run the labs step by step for better understanding, following the provided instructions closely.

The challenges of training large language models (LLMs) due to memory limitations and introduces quantization as a solution to reduce memory requirements.

Memory Challenges in Training LLMs

Large language models require significant GPU memory, often leading to out-of-memory errors during training.
For example, a one billion parameter model at 32-bit precision requires approximately 24 GB of GPU RAM, which is often unfeasible for consumer hardware.
Quantization as a Solution

Quantization reduces the precision of model weights from 32-bit to lower bit representations (16-bit or 8-bit), significantly decreasing memory usage.
For instance, using 16-bit floating point (FP16) reduces memory requirements by half, while 8-bit integers (INT8) can reduce it further to one-fourth.
Types of Quantization

BFLOAT16 is highlighted as a popular choice in deep learning, maintaining the dynamic range of FP32 while reducing memory footprint.
The trade-off with quantization is a loss of precision, but this is often acceptable for many applications.
Overall Impact of Quantization

By applying quantization, memory consumption can be drastically reduced, allowing for the training of larger models.
As model sizes increase beyond a few billion parameters, distributed computing techniques become necessary for training, which can be costly and complex.

Strategies for scaling model training across multiple GPUs, focusing on two main techniques: Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP).

Distributed Data Parallel (DDP)

DDP replicates the model across multiple GPUs, sending batches of data to each GPU for parallel processing.
It requires that model weights and training parameters fit on a single GPU, making it suitable for smaller models.
Fully Sharded Data Parallel (FSDP)

FSDP addresses memory limitations by sharding model parameters, gradients, and optimizer states across GPUs, reducing redundancy.
It utilizes the ZeRO technique, which offers three optimization stages to minimize memory usage, allowing for training larger models that exceed single GPU memory limits.
In summary, both DDP and FSDP provide methods to efficiently utilize multiple GPUs for model training, with FSDP being particularly advantageous for larger models.

The computational challenges and considerations in training large language models (LLMs).

Model Size and Performance

The goal during pre-training is to maximize model performance by minimizing loss when predicting tokens.
Increasing the dataset size or the number of model parameters can improve performance, but compute budget constraints must also be considered.
Compute Budget and Measurement

A petaFLOP per second day quantifies the compute resources needed, equivalent to one quadrillion floating point operations per second for a day.
Training larger models like GPT-3 requires significantly more compute resources, with the largest models needing thousands of petaFLOP per second days.
Scaling Relationships

Research shows a power-law relationship between compute budget, model size, and training dataset size, indicating that increasing compute generally leads to better performance.
The Chinchilla paper suggests that optimal training dataset size should be about 20 times the number of model parameters for effective training.

The importance of pretraining large language models (LLMs) for specific domains where common vocabulary may not suffice.

Domain-Specific Pretraining

Pretraining is essential when working with specialized vocabulary, such as legal or medical terms, which may not be well-represented in existing LLMs.
Examples include legal terms like "mens rea" and "res judicata," which are rarely used outside the legal field, making it difficult for models to understand them.
Challenges in Specialized Language

Medical language often includes uncommon words and shorthand that may not appear in general training datasets, leading to potential misinterpretations.
The need for domain adaptation arises to ensure models perform well in specialized contexts.
Case Study: BloombergGPT

BloombergGPT is a pretrained model specifically for finance, combining financial and general-purpose data to achieve high performance on financial benchmarks.
The model's training involved trade-offs due to limited financial data, highlighting real-world constraints in model development.
Overall Recap of Week 1

The week covered common LLM use cases, transformer architecture, training challenges, and scaling laws for optimal model design.

BloombergGPT is a specialized language model designed for finance, developed by Bloomberg. 

Pre-training and Dataset

It was pre-trained on a large financial dataset, including news articles, reports, and market data, to enhance its finance-related text generation capabilities.
The training utilized Chinchilla Scaling Laws to determine the optimal number of parameters and training data volume, aiming for 50 billion parameters and 1.4 trillion tokens.
Training Challenges

The team faced difficulties in acquiring the full 1.4 trillion tokens, resulting in a dataset of 700 billion tokens, which is below the compute-optimal value.
The training process was further limited by early stopping, concluding after processing 569 billion tokens.
Domain-Specific Pre-training

BloombergGPT exemplifies the process of pre-training a model for increased specificity in a particular domain, highlighting the trade-offs that can occur when optimal configurations are not achievable.

Instruction tuning and fine-tuning of large language models (LLMs) to enhance their performance and adaptability.

Instruction Tuning

Instruction fine-tuning helps pretrained models learn to respond effectively to specific prompts and tasks.
It is a significant advancement in LLMs, allowing them to follow instructions rather than just predict the next word based on general text.
Fine-Tuning Techniques

Two main types of fine-tuning are discussed: instruction fine-tuning and application-specific fine-tuning.
Parameter Efficient Fine-Tuning (PEFT) methods help reduce the computational and memory costs associated with full fine-tuning.
Challenges and Solutions

Catastrophic forgetting can occur during fine-tuning, where the model loses previously learned information.
Techniques like LoRA (Low-Rank Adaptation) are highlighted for their efficiency in achieving good performance with lower resource requirements.

Fine-tuning large language models (LLMs) to enhance their performance for specific tasks.

Fine-Tuning Overview

Fine-tuning is a supervised learning process that updates the weights of a base model using a dataset of labeled examples.
Instruction fine-tuning is a method that improves a model's performance across various tasks by training it with examples that demonstrate how to respond to specific instructions.
Preparing Training Data

To fine-tune a model, you need to prepare a dataset of prompt-completion pairs, where each prompt includes an instruction.
Prompt template libraries can help convert existing datasets into instruction prompt datasets suitable for fine-tuning.
Fine-Tuning Process

The fine-tuning process involves dividing the dataset into training, validation, and test splits, and then comparing the model's output with the expected responses.
The model's weights are updated through backpropagation based on the calculated loss, improving its performance on the task.
Evaluation and Results

After fine-tuning, you evaluate the model's performance using validation and test datasets to assess its accuracy.
The outcome is a new version of the model, often referred to as an instruct model, which is better suited for the specific tasks of interest.
