//! OpenRouter API client for embeddings
//!
//! Uses OpenAI-compatible embedding endpoint via OpenRouter.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

/// OpenRouter client configuration
#[derive(Clone)]
pub struct OpenRouterClient {
    api_key: String,
    base_url: String,
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
    /// Create a new client with the given API key
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            model: "openai/text-embedding-3-small".to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Create client from environment variable or clawdbot.json
    ///
    /// Resolution order:
    /// 1. OPENROUTER_API_KEY environment variable
    /// 2. ~/.clawdbot/clawdbot.json (models.providers.openrouter.apiKey)
    pub fn from_env() -> Result<Self> {
        // Try env var first (for standalone/public use)
        if let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") {
            return Ok(Self::new(api_key));
        }

        // Fall back to clawdbot.json (for integrated Clawdbot users)
        let api_key = Self::read_from_clawdbot_config()
            .context("OPENROUTER_API_KEY not set and not found in ~/.clawdbot/clawdbot.json")?;

        Ok(Self::new(api_key))
    }

    /// Read API key from clawdbot.json
    fn read_from_clawdbot_config() -> Result<String> {
        let home = std::env::var("HOME").context("HOME not set")?;
        let config_path = std::path::Path::new(&home).join(".clawdbot/clawdbot.json");

        let content = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;

        let config: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse clawdbot.json")?;

        config
            .get("models")
            .and_then(|m| m.get("providers"))
            .and_then(|p| p.get("openrouter"))
            .and_then(|o| o.get("apiKey"))
            .and_then(|k| k.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("OpenRouter API key not found in clawdbot.json"))
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
                .post(format!("{}/embeddings", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
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
