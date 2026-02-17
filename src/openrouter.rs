//! API clients for embeddings (OpenAI) and chat (OpenRouter).
//!
//! Embeddings go direct to OpenAI (ZDR-compatible, same model).
//! Chat completions (harvest) go through OpenRouter for model variety.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// API client — uses OpenAI direct for embeddings, OpenRouter for chat.
#[derive(Clone)]
pub struct OpenRouterClient {
    /// OpenRouter key (for chat/harvest)
    api_key: String,
    base_url: String,
    /// OpenAI key (for embeddings)
    openai_api_key: String,
    openai_base_url: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Option<EmbeddingUsage>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct EmbeddingUsage {
    prompt_tokens: Option<i32>,
    total_tokens: Option<i32>,
}

/// Chat completion request for harvest
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

impl OpenRouterClient {
    /// Create a new client with separate keys for embeddings (OpenAI) and chat (OpenRouter).
    pub fn new(openrouter_key: String, openai_key: String) -> Self {
        Self {
            api_key: openrouter_key,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            openai_api_key: openai_key,
            openai_base_url: "https://api.openai.com/v1".to_string(),
            model: "text-embedding-3-small".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create client from environment variables or clawdbot.json
    ///
    /// Embedding key resolution (OpenAI):
    /// 1. OPENAI_API_KEY environment variable
    /// 2. ~/.clawdbot/clawdbot.json (models.providers.openai.apiKey)
    ///
    /// Chat key resolution (OpenRouter):
    /// 1. OPENROUTER_API_KEY environment variable
    /// 2. ~/.clawdbot/clawdbot.json (models.providers.openrouter.apiKey)
    pub fn from_env() -> Result<Self> {
        let openai_key = std::env::var("OPENAI_API_KEY")
            .or_else(|_| Self::read_key_from_config("openai"))
            .context("OPENAI_API_KEY not set and not found in clawdbot.json")?;

        let openrouter_key = std::env::var("OPENROUTER_API_KEY")
            .or_else(|_| Self::read_key_from_config("openrouter"))
            .context("OPENROUTER_API_KEY not set and not found in clawdbot.json")?;

        Ok(Self::new(openrouter_key, openai_key))
    }

    /// Read an API key from clawdbot.json for the given provider
    fn read_key_from_config(provider: &str) -> Result<String> {
        let home = std::env::var("HOME").context("HOME not set")?;

        // Try both config paths (openclaw and clawdbot)
        let candidates = [
            std::path::PathBuf::from(&home).join(".openclaw/openclaw.json"),
            std::path::PathBuf::from(&home).join(".clawdbot/clawdbot.json"),
        ];

        for config_path in &candidates {
            if let Ok(content) = std::fs::read_to_string(config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(key) = config
                        .get("models")
                        .and_then(|m| m.get("providers"))
                        .and_then(|p| p.get(provider))
                        .and_then(|o| o.get("apiKey"))
                        .and_then(|k| k.as_str())
                    {
                        return Ok(key.to_string());
                    }
                }
            }
        }

        Err(anyhow!("{} API key not found in config files", provider))
    }

    /// Set the embedding model
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Generate embeddings for a batch of texts
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // OpenRouter has a limit on batch size, process in chunks
        const BATCH_SIZE: usize = 100;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let request = EmbeddingRequest {
                model: self.model.clone(),
                input: chunk.to_vec(),
            };

            let response = self
                .client
                .post(format!("{}/embeddings", self.openai_base_url))
                .header("Authorization", format!("Bearer {}", self.openai_api_key))
                .header("Content-Type", "application/json")
                .json(&request)
                .send()
                .await
                .context("Failed to send embedding request")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Embedding API error {}: {}", status, body));
            }

            let result: EmbeddingResponse = response
                .json()
                .await
                .context("Failed to parse embedding response")?;

            // Sort by index to maintain order
            let mut embeddings: Vec<_> = result.data.into_iter().collect();
            embeddings.sort_by_key(|e| e.index);

            all_embeddings.extend(embeddings.into_iter().map(|e| e.embedding));
        }

        Ok(all_embeddings)
    }

    /// Generate embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding returned"))
    }

    /// Call chat completion for harvesting (uses cheaper model)
    pub async fn chat(&self, system_prompt: &str, user_prompt: &str, model: &str) -> Result<String> {
        let request = ChatRequest {
            model: model.to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            temperature: 0.3,
        };

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send chat request")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("Chat API error {}: {}", status, body));
        }

        let result: ChatResponse = response
            .json()
            .await
            .context("Failed to parse chat response")?;

        result
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("No response from chat API"))
    }
}
