# llama31_prompt_guard_rs

Llama 3.1 Prompt Guard 86M with Rust and ONNX runtime

# Setup

1. Get access to model at https://huggingface.co/meta-llama/Prompt-Guard-86M

2. Using Huggingface to export model to ONNX format

https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model

3. Put files in `/model`

4. Install onnx `brew install onnxruntime`

5. export env vars:

`export ORT_INCLUDE_DIR=/opt/homebrew/Cellar/onnxruntime/1.17.1/include`

`export ORT_LIB_LOCATION=opt/homebrew/Cellar/onnxruntime/1.17.1/lib`

`export ORT_STRATEGY=system`

6. Run example with `cargo run`
