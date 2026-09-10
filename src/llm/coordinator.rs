// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Detecting whether an endpoint is an orangu-coordinator proxy rather than a
//! plain orangu-server — shared by both the `orangu`
//! binary (the header status probe, startup embeddings detection, `/server`
//! re-detection) and library code that needs it without access to any of
//! that (the explorer subagent, which reloads its own config from disk).

use crate::llm::normalized_openai_endpoint;
use serde_json::Value;

/// `GET /v1/coordinator`: `Some(models)` when the endpoint confirms itself as
/// an orangu-coordinator proxy (`"orangu_coordinator": true`), carrying every
/// distinct model each conventional role (`all`/`code`/`review`/`explorer`/
/// `embeddings`) currently resolves to, deduplicated; `None` for anything
/// else — unreachable, a non-success status, an unexpected body, or a plain
/// orangu-server, neither of which exposes this path.
pub async fn probe_coordinator(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> Option<Vec<String>> {
    let url = format!("{}/v1/coordinator", normalized_openai_endpoint(endpoint));
    let mut request = http_client.get(url);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    parse_coordinator_body(&body)
}

/// The roles an orangu-coordinator at `endpoint` has a **configured profile**
/// for (`GET /v1/coordinator`'s `roles`), or `None` when the endpoint is not a
/// coordinator or is too old to report them.
///
/// The point of asking is to learn what a coordinator can do *without making
/// it do it*. Establishing embeddings support by sending an embeddings request
/// is a model swap: the coordinator stops the chat server, loads the embedding
/// model, and the next request loads the chat model back — twice the load time
/// of a cold start and a discarded KV cache, at every client launch, to answer
/// a question the coordinator can simply state.
pub async fn probe_coordinator_roles(
    http_client: &reqwest::Client,
    endpoint: &str,
    api_key: Option<&str>,
) -> Option<Vec<String>> {
    let url = format!("{}/v1/coordinator", normalized_openai_endpoint(endpoint));
    let mut request = http_client.get(url);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    parse_coordinator_roles(&body)
}

/// Parses `roles` out of a `GET /v1/coordinator` body, or `None` when the body
/// is not a coordinator's or carries no `roles` — an older coordinator, whose
/// caller then falls back to establishing the capability the expensive way.
fn parse_coordinator_roles(body: &Value) -> Option<Vec<String>> {
    if !body
        .get("orangu_coordinator")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    let roles: Vec<String> = body
        .get("roles")?
        .as_array()?
        .iter()
        .filter_map(|role| role.as_str().map(str::to_string))
        .collect();
    (!roles.is_empty()).then_some(roles)
}

/// Parses a `GET /v1/coordinator` response body: `Some(models)` — every
/// distinct model named in its `models` map, deduplicated — when
/// `orangu_coordinator` is `true`, `None` otherwise. Split out of
/// [`probe_coordinator`] so the parsing itself is testable without a live
/// server.
fn parse_coordinator_body(body: &Value) -> Option<Vec<String>> {
    if !body
        .get("orangu_coordinator")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }

    let mut models: Vec<String> = body
        .get("models")
        .and_then(Value::as_object)
        .map(|roles| {
            roles
                .values()
                .filter_map(|model| model.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    models.sort_unstable();
    models.dedup();
    Some(models)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_coordinator_body_reads_and_dedupes_models() {
        let body = serde_json::json!({
            "orangu_coordinator": true,
            "version": "0.11.0",
            "models": {
                "all": "org/gemma",
                "code": "org/gemma",
                "explorer": "org/qwen",
            },
        });
        let mut models = super::parse_coordinator_body(&body).expect("is a coordinator");
        models.sort_unstable();
        assert_eq!(
            models,
            vec!["org/gemma".to_string(), "org/qwen".to_string()]
        );
    }

    /// The field exists so a client can learn what a coordinator can do
    /// without making it do it — an embeddings *request* through a
    /// coordinator is a model swap, not a question.
    #[test]
    fn roles_are_read_only_from_a_coordinators_own_answer() {
        use serde_json::json;
        let body = json!({
            "orangu_coordinator": true,
            "models": {"all": "org/chat", "embeddings": "org/chat"},
            "roles": ["all", "code", "embeddings"],
        });
        assert_eq!(
            super::parse_coordinator_roles(&body),
            Some(vec![
                "all".to_string(),
                "code".to_string(),
                "embeddings".to_string()
            ])
        );

        // An older coordinator names no roles: `None`, so the caller falls
        // back to establishing the capability rather than concluding it has
        // none.
        let older = json!({"orangu_coordinator": true, "models": {"all": "org/chat"}});
        assert_eq!(super::parse_coordinator_roles(&older), None);

        // And a plain server's body is never read as a coordinator's.
        let plain = json!({"roles": ["all"]});
        assert_eq!(super::parse_coordinator_roles(&plain), None);
    }

    #[test]
    fn parse_coordinator_body_rejects_non_coordinator_responses() {
        assert_eq!(super::parse_coordinator_body(&serde_json::json!({})), None);
        assert_eq!(
            super::parse_coordinator_body(&serde_json::json!({"status": "ok"})),
            None
        );
        assert_eq!(
            super::parse_coordinator_body(&serde_json::json!({"orangu_coordinator": false})),
            None
        );
    }
}
