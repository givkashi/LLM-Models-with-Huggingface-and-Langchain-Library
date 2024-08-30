# LLM-HF-LangChain: A Comprehensive Guide to Exploring LLM Models with Hugging Face and LangChain <a href="https://medium.com/@givkashi/exploring-llm-models-with-hugging-face-and-langchain-library-on-google-colab-a-comprehensive-guide-4994e7ed5c06" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/medium.svg" alt="@_giaabaoo_" height="30" width="40" /></a>

<center>
    <a href="https://github.com/givkashi/LLM-Models-with-Huggingface-and-Langchain-Library/blob/main/LLM_HF_LangChain.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</center>
<img src="https://github.com/givkashi/LLM-Models-with-Huggingface-and-Langchain-Library/blob/main/img.png" width="60%" height="60%"/>

## Overview

This repository provides a detailed guide for exploring different Large Language Models (LLMs) available on Hugging Face using the LangChain library. The notebook demonstrates how to set up, initialize, and interact with models like **Llama**, **Mistral**, and **Phi**. The code offers a step-by-step approach to configure the environment, initialize models, and test their performance using prompt templates and pipelines.

## Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Model Initialization](#model-initialization)
- [Generation Configuration](#generation-configuration)
- [Pipeline Creation](#pipeline-creation)
- [Model Testing](#model-testing)
- [Advanced Testing with PromptTemplate](#advanced-testing-with-prompttemplate)
- [Advanced Testing with ChatPromptTemplate](#advanced-testing-with-chatprompttemplate)

## Installation

Before running the notebook, you need to install the required Python libraries. The following command will install the necessary dependencies:

```bash
pip install -q -U langchain transformers bitsandbytes accelerate
```

## Environment Setup

To start exploring the LLM models, you first need to set up your environment. This involves configuring your Hugging Face API key and importing the required libraries.

```python
import torch
import os
from langchain import PromptTemplate, HuggingFacePipeline
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

os.environ["HF_TOKEN"]='your_huggingface_API_key'
```

## Model Initialization

You can explore various LLM models by initializing them using Hugging Face's model hub. The repository provides a way to configure the model with quantization for efficient memory and computation usage.

```python
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=quantization_config
)
```

## Generation Configuration

The generation settings define how the model generates text. This section provides an example configuration:

```python
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.7
generation_config.top_p = 0
generation_config.do_sample = True
```

## Pipeline Creation

The pipeline connects the tokenizer, model, and generation configuration, providing an API for text generation.

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
)

llm = HuggingFacePipeline(pipeline=pipe)
```

## Model Testing

You can test the model by providing input text and generating outputs.

```python
input_text = "Write me a poem about Machine Learning."

output = llm.invoke(input_text)
print(output)
```

## Advanced Testing with PromptTemplate

The repository demonstrates how to use `PromptTemplate` to create more structured inputs for the model.

```python
template = """
     Write me a poem about {topic}.
"""

topic = "Machine Learning"

prompt = PromptTemplate(input_variables=["topic"], template=template)
chain = prompt | llm
output = chain.invoke({"topic": topic})
print(output)
```

## Advanced Testing with ChatPromptTemplate

This section shows how to create a conversational model interaction using `ChatPromptTemplate`.

```python
topic = "Machine Learning"

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("Write a poem related to the input topic in one paragraph")),
        HumanMessagePromptTemplate.from_template("```{topic}```"),
    ]
)

chain = prompt | llm
output = chain.invoke({"topic": topic})
print(output)
```

