import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

import streamlit as st


def generate_text_by_open_calm(input_text: str) -> str:
    """
    Generate text by OpenCALM

    Parameters
    ----------
    input_text: str

    Returns
    -------
    output: str
    """
    model_name = "cyberagent/open-calm-small"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    #     model.to(device)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output


def generate_text_by_rinna_gpt_neox(input_text: str) -> str:
    """
    Generate text by Rinna GPT-NeoX

    Parameters
    ----------
    input_text: str

    Returns
    -------
    output: str
    """
    model_name = "rinna/japanese-gpt-neox-small"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False)

    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')
    #     model.to(device)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output


def generate_text_by_line_jpn_llm(input_text: str) -> str:
    """
    Generate text by LINE japanese large lm

    Parameters
    ----------
    input_text: str

    Returns
    -------
    output: str
    """
    model_name = "line-corporation/japanese-large-lm-1.7b"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=model.device)

    outputs = generator(
        input_text,
        max_length=64,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )

    return outputs[0]["generated_text"]


if __name__ == "__main__":
    st.title("JPN LLM")

    model_names = ["OpenCALM", "Rinna GPT-NeoX", "LINE JPN LLM"]

    with st.sidebar:
        selected_model = st.selectbox(
            "model",
            model_names
        )

    input_text = st.text_input("input:")

    if input_text != "":
        output = ""

        if selected_model == model_names[0]:
            output = generate_text_by_open_calm(input_text)
        elif selected_model == model_names[1]:
            output = generate_text_by_rinna_gpt_neox(input_text)
        elif selected_model == model_names[2]:
            output = generate_text_by_line_jpn_llm(input_text)

        print(output)

        st.text_area(f"output by {selected_model}:", output, height=300)
