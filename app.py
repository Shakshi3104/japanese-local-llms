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


def generate_text_by_line_jpn_llm_sft(input_text: str) -> str:
    """

    Parameters
    ----------
    input_text: str

    Returns
    -------
    output: str
    """
    model_name = "line-corporation/japanese-large-lm-1.7b-instruction-sft"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=False)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=model.device)

    outputs = generator(
        f"ユーザー: {input_text}\nシステム: ",
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )

    return outputs[0]["generated_text"].replace(f"ユーザー: {input_text}\nシステム: ", "")


def generate_text_by_line_jpn_llm(input_text: str, fine_tuned=False) -> str:
    """
    Generate text by LINE japanese large lm

    Parameters
    ----------
    input_text: str
    fine_tuned: bool

    Returns
    -------
    output: str
    """
    # Supervised Fine-tuning by Instruction Tuning
    if fine_tuned:
        return generate_text_by_line_jpn_llm_sft(input_text)

    # else: vanilla LLM
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

    # select the model
    with st.sidebar:
        selected_model = st.selectbox(
            "model",
            model_names
        )

    selected_model_name = selected_model
    input_text = st.text_input("input:")

    # LINE's options
    options = ["Vanilla", "Tuned"]
    if selected_model == model_names[2]:
        with st.sidebar:
            selected_option = st.radio(
                "option",
                options
            )

    if input_text != "":
        output = ""

        if selected_model == model_names[0]:
            # OpenCALM by CyberAgent
            output = generate_text_by_open_calm(input_text)
        elif selected_model == model_names[1]:
            # GPT-NeoX by Rinna
            output = generate_text_by_rinna_gpt_neox(input_text)
        elif selected_model == model_names[2]:
            # Japanese large LM by LINE
            if selected_option == options[0]:
                selected_model_name = f"{selected_model}"
                output = generate_text_by_line_jpn_llm(input_text)
            elif selected_option == options[1]:
                selected_model_name = f"{selected_model} (Instruction Tuning)"
                output = generate_text_by_line_jpn_llm(input_text, fine_tuned=True)

        print(output)

        st.text_area(f"output by {selected_model_name}:", output, height=300)
