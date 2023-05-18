import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import streamlit as st


if __name__ == "__main__":
    st.title("OpenCALM")

    # print(torch.backends.mps.is_available())
    # device = torch.device('mps')

    model_name = "cyberagent/open-calm-small"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = st.text_input("input:")

    if input_text != "":
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
        print(output)

        st.write(output)
