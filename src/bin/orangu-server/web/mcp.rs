// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! The MCP inventory for the web console: the servers the configuration
//! file names, listed, and rewritten from the console's Settings › MCP
//! pane. The server itself neither connects to an MCP server nor calls
//! one — these sections are an inventory for orangu clients — so an edit
//! is a change to the file: it is written at once, and a server started
//! from the file afterwards sees it. `PUT /api/mcps` answers with where it
//! wrote and that a restart is what applies it, which the console shows.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::WebState;
use crate::config::McpConfiguration;

pub fn router() -> Router<Arc<WebState>> {
    Router::new()
        .route("/api/mcps", get(list).put(replace))
        .route("/api/mcps/{name}", get(show))
}

#[derive(Serialize, Deserialize, Clone)]
struct McpView {
    name: String,
    endpoint: String,
    #[serde(default = "enabled_default")]
    enabled: bool,
    #[serde(default = "approval_default")]
    approval_mode: String,
}

fn enabled_default() -> bool {
    true
}

fn approval_default() -> String {
    "writes".to_string()
}

fn view(mcp: &McpConfiguration) -> McpView {
    McpView {
        name: mcp.name.clone(),
        endpoint: mcp.endpoint.clone(),
        enabled: mcp.enabled,
        approval_mode: mcp.approval_mode.clone(),
    }
}

/// The list, with where it comes from: `path` is the configuration file
/// the pane's Save writes, `null` when there is none — a bundled binary
/// running on its built-in answers — in which case the pane is read-only.
#[derive(Serialize)]
struct Inventory {
    servers: Vec<McpView>,
    path: Option<String>,
    /// Set on the answer to a `PUT`: what was written is in the file, and
    /// a server started from it is what reads it.
    restart_required: bool,
}

fn inventory(state: &WebState, restart_required: bool) -> Inventory {
    let servers = state
        .mcp_servers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Inventory {
        servers: servers.iter().map(view).collect(),
        path: state.config_file.as_ref().map(|p| p.display().to_string()),
        restart_required,
    }
}

async fn list(State(state): State<Arc<WebState>>) -> Json<Inventory> {
    Json(inventory(&state, false))
}

async fn show(State(state): State<Arc<WebState>>, Path(name): Path<String>) -> impl IntoResponse {
    let servers = state
        .mcp_servers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match servers.iter().find(|mcp| mcp.name == name) {
        Some(mcp) => Json(view(mcp)).into_response(),
        None => (StatusCode::NOT_FOUND, "MCP server not found").into_response(),
    }
}

/// `PUT /api/mcps` — the whole inventory as the pane holds it after its
/// adds, edits and deletes. Every entry is checked before the file is
/// touched, so a refused one leaves the file as it was; then each section
/// the file no longer names is removed, and each that is new or changed
/// is rewritten — the unchanged ones keep their lines, comments and all.
async fn replace(
    State(state): State<Arc<WebState>>,
    Json(wanted): Json<Vec<McpView>>,
) -> axum::response::Response {
    let Some(path) = state.config_file.clone() else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this server has no configuration file to write MCP servers to",
        )
            .into_response();
    };
    let wanted: Vec<McpConfiguration> = wanted
        .into_iter()
        .map(|v| McpConfiguration {
            name: v.name.trim().to_string(),
            endpoint: v.endpoint.trim().to_string(),
            enabled: v.enabled,
            approval_mode: v.approval_mode.trim().to_string(),
        })
        .collect();
    for mcp in &wanted {
        if let Err(err) = mcp.validate() {
            return (StatusCode::BAD_REQUEST, err.to_string()).into_response();
        }
    }
    for (i, mcp) in wanted.iter().enumerate() {
        if wanted[..i].iter().any(|other| other.name == mcp.name) {
            return (
                StatusCode::BAD_REQUEST,
                format!("two MCP servers named '{}'", mcp.name),
            )
                .into_response();
        }
    }
    // What the file holds now — not what this process came up with, in
    // case it was edited by hand since — is what the diff is against.
    let current = match crate::config::load_mcp_servers(&path) {
        Ok(current) => current,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response(),
    };
    let same = |a: &McpConfiguration, b: &McpConfiguration| {
        a.endpoint == b.endpoint && a.enabled == b.enabled && a.approval_mode == b.approval_mode
    };
    let mut result = Ok(());
    for old in &current {
        if !wanted.iter().any(|mcp| mcp.name == old.name) {
            result = crate::config::rewrite_mcp_section(&path, &old.name, None);
            if result.is_err() {
                break;
            }
        }
    }
    if result.is_ok() {
        for mcp in &wanted {
            let unchanged = current
                .iter()
                .any(|old| old.name == mcp.name && same(old, mcp));
            if unchanged {
                continue;
            }
            result = crate::config::rewrite_mcp_section(&path, &mcp.name, Some(mcp));
            if result.is_err() {
                break;
            }
        }
    }
    if let Err(err) = result {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response();
    }
    // The list the console shows is the file's, read back rather than
    // assumed, so what it displays is what a restart will read.
    match crate::config::load_mcp_servers(&path) {
        Ok(servers) => {
            log::info!(
                "orangu-server: MCP servers written to {} ({} configured) — a restart reads them",
                path.display(),
                servers.len()
            );
            *state
                .mcp_servers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = servers;
            Json(inventory(&state, true)).into_response()
        }
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("{err:#}")).into_response(),
    }
}
