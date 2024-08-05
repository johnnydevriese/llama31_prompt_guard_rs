use ndarray::Array2;
use onnxruntime::{environment::Environment, session::Session, tensor::OrtOwnedTensor};
use tokenizers::Tokenizer;

use serde_json::Value;
use std::fs::File;
use std::io::Read;
use tokenizers::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy};
use tokenizers::AddedToken;

// NOTE: The following environment variables must be set for the onnxruntime crate to work:
// TODO: use dotenv instead of exporting each time.
// ORT_INCLUDE_DIR   -- /opt/homebrew/Cellar/onnxruntime/1.17.1/include
// ORT_LIB_LOCATION  -- /opt/homebrew/Cellar/onnxruntime/1.17.1/lib
// ORT_STRATEGY      -- system

fn load_tokenizer(
    tokenizer_path: &str,
    special_tokens_map_path: &str,
    tokenizer_config_path: &str,
) -> Tokenizer {
    // Load the tokenizer
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

    // Load and apply special tokens
    let mut special_tokens_file = File::open(special_tokens_map_path).unwrap();
    let mut special_tokens_contents = String::new();
    special_tokens_file
        .read_to_string(&mut special_tokens_contents)
        .unwrap();
    let special_tokens: Value = serde_json::from_str(&special_tokens_contents).unwrap();

    // Apply special tokens
    let mut special_tokens_to_add = Vec::new();
    if let Some(cls_token) = special_tokens["cls_token"].as_str() {
        special_tokens_to_add.push(AddedToken::from(cls_token, true));
    }
    if let Some(sep_token) = special_tokens["sep_token"].as_str() {
        special_tokens_to_add.push(AddedToken::from(sep_token, true));
    }
    if let Some(pad_token) = special_tokens["pad_token"].as_str() {
        special_tokens_to_add.push(AddedToken::from(pad_token, true));
    }
    tokenizer.add_special_tokens(&special_tokens_to_add);

    // Load tokenizer config
    let mut config_file = File::open(tokenizer_config_path).unwrap();
    let mut config_contents = String::new();
    config_file.read_to_string(&mut config_contents).unwrap();
    let config: Value = serde_json::from_str(&config_contents).unwrap();

    // Apply relevant config settings
    let max_length = config["model_max_length"].as_u64().unwrap_or(512) as usize;
    let padding_side = config["padding_side"].as_str().unwrap_or("right");
    let pad_token = config["pad_token"].as_str().unwrap_or("[PAD]");
    let pad_id = tokenizer.token_to_id(pad_token).unwrap_or(0);
    let pad_type_id = config["pad_token_type_id"].as_u64().unwrap_or(0) as u32;

    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(max_length),
        direction: if padding_side == "right" {
            PaddingDirection::Right
        } else {
            PaddingDirection::Left
        },
        pad_to_multiple_of: None, // Set this to Some(value) if needed
        pad_id: pad_id as u32,
        pad_type_id,
        pad_token: pad_token.to_string(),
    }));

    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length,
            ..Default::default()
        }))
        .unwrap();

    tokenizer
}

fn softmax(x: &[f32], temperature: f32) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|&x| ((x - max) / temperature).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

struct OnnxSequenceClassificationModel<'a> {
    session: Session<'a>,
}

impl<'a> OnnxSequenceClassificationModel<'a> {
    fn new(environment: &'a Environment, model_path: &'a str) -> Self {
        let session = environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(model_path)
            .unwrap();

        OnnxSequenceClassificationModel { session }
    }

    fn predict(&mut self, input_ids: Vec<i64>, attention_mask: Vec<i64>) -> Vec<f32> {
        let input_ids = Array2::from_shape_vec((1, input_ids.len()), input_ids).unwrap();
        let attention_mask =
            Array2::from_shape_vec((1, attention_mask.len()), attention_mask).unwrap();

        let inputs = vec![input_ids.into_dyn(), attention_mask.into_dyn()];

        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(inputs).unwrap();
        outputs[0].view().to_owned().as_slice().unwrap().to_vec()
    }
}

fn get_class_probabilities(
    model: &mut OnnxSequenceClassificationModel,
    tokenizer: &Tokenizer,
    text: &str,
    temperature: f32,
) -> Vec<f32> {
    let encoding = tokenizer.encode(text, true).unwrap();
    let input_ids = encoding
        .get_ids()
        .iter()
        .map(|&id| id as i64)
        .collect::<Vec<_>>();
    let attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|&mask| mask as i64)
        .collect::<Vec<_>>();

    let logits = model.predict(input_ids, attention_mask);
    println!("Logits: {:?}", logits);
    softmax(&logits, temperature)
}

fn get_jailbreak_score(
    model: &mut OnnxSequenceClassificationModel,
    tokenizer: &Tokenizer,
    text: &str,
    temperature: f32,
) -> f32 {
    let probabilities = get_class_probabilities(model, tokenizer, text, temperature);
    println!("Probabilities: {:?}", probabilities);
    probabilities[2] // Return the probability for the "JAILBREAK" class
}

fn get_indirect_injection_score(
    model: &mut OnnxSequenceClassificationModel,
    tokenizer: &Tokenizer,
    text: &str,
    temperature: f32,
) -> f32 {
    let probabilities = get_class_probabilities(model, tokenizer, text, temperature);
    println!("Probabilities: {:?}", probabilities);
    probabilities[1] + probabilities[2] // Sum of "INJECTION" and "JAILBREAK" probabilities
}

fn get_benign_score(
    model: &mut OnnxSequenceClassificationModel,
    tokenizer: &Tokenizer,
    text: &str,
    temperature: f32,
) -> f32 {
    let probabilities = get_class_probabilities(model, tokenizer, text, temperature);
    println!("Probabilities: {:?}", probabilities);
    probabilities[0] // Return the probability for the "BENIGN" class
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let environment = Environment::builder()
        .with_name("onnx_environment")
        .build()
        .unwrap();

    let tokenizer = load_tokenizer(
        "./model/tokenizer.json",
        "./model/special_tokens_map.json",
        "./model/tokenizer_config.json",
    );

    let model_path = "./model/model.onnx";
    let mut model = OnnxSequenceClassificationModel::new(&environment, model_path);

    let input_text = "hello world!";
    let temperature = 1.0;

    println!("Input text: {}", input_text);
    let benign_score = get_benign_score(&mut model, &tokenizer, input_text, temperature);
    println!("Benign score: {:.4}", benign_score);

    let jailbreak_score = get_jailbreak_score(&mut model, &tokenizer, input_text, temperature);
    println!("Jailbreak score: {:.4}", jailbreak_score);

    let indirect_injection_score =
        get_indirect_injection_score(&mut model, &tokenizer, input_text, temperature);
    println!("Indirect injection score: {:.4}", indirect_injection_score);

    Ok(())
}
