import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import streamlit as st


def generate_text_by_open_calm(input_text: str) -> str:

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


if __name__ == "__main__":
    st.title("JPN LLM")

    # print(torch.backends.mps.is_available())
    # device = torch.device('mps')

    model_names = ["OpenCALM", "Rinna GPT-NeoX"]
    model_types = ["cyberagent/open-calm-small",
                   "rinna/japanese-gpt-neox-small"]

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

        print(output)

        st.text_area(f"output by {selected_model}:", output, height=300)
