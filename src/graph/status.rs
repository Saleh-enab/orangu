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

/// Build status of a workspace's knowledge graph — the background scan that
/// powers `graph_lookup`, `/graph`, and Deep `/auto_review`'s cross-file
/// context. A caller holds this behind an `Arc<Mutex<GraphBuildStatus>>`
/// (see `ToolExecutor::graph_status`) alongside the graph itself
/// (`ToolExecutor::graph_store`), updating it from `Building` once when the
/// scan starts, then to `Ready` or `Failed` once it ends.
///
/// This is a UI-facing signal, not a correctness gate: every graph query
/// already tolerates a graph that isn't built yet (e.g.
/// `auto_review_graph_context` and the `graph_lookup` tool both treat a
/// `None` store as "nothing found" rather than an error) — `GraphBuildStatus`
/// exists so that behavior can be *surfaced* (a status-bar dot, a warning)
/// instead of silently read as "no results."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphBuildStatus {
    /// The scan hasn't finished yet: every graph query comes back empty.
    #[default]
    Building,
    /// The scan completed and the graph is populated.
    Ready,
    /// The scan task itself failed (panicked or was cancelled) — the graph
    /// will stay empty for the rest of the session.
    Failed,
}

/// How many workspace scans are running right now, shared by everyone who
/// starts one (the startup scan in `orangu`'s event loop) and read by anyone
/// who would rather wait for a graph than go without it.
///
/// It answers the one question [`GraphBuildStatus`] cannot: `Building` is the
/// default, so it reads the same whether a scan is under way or none was ever
/// started (a `-p "/graph"` one-shot starts none). A waiter needs to tell
/// those apart — the first is worth waiting for, the second means the waiter
/// has to scan the workspace itself.
///
/// A scan raises the count with [`ScanActivity::begin`] and lowers it when the
/// returned guard drops, so a scan that panics or is cancelled still releases
/// its waiters. A scan must publish its store *before* its guard drops: a
/// waiter that sees nothing scanning takes the store as final.
#[derive(Clone, Debug, Default)]
pub struct ScanActivity(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl ScanActivity {
    /// Registers a scan as running until the returned guard is dropped.
    pub fn begin(&self) -> ScanGuard {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ScanGuard(self.0.clone())
    }

    /// Whether at least one scan is running.
    pub fn is_scanning(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst) > 0
    }
}

/// Keeps a [`ScanActivity`] count raised for as long as it lives.
pub struct ScanGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for ScanGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_activity_tracks_running_scans() {
        let activity = ScanActivity::default();
        assert!(!activity.is_scanning());

        let outer = activity.begin();
        assert!(activity.is_scanning());
        {
            let _inner = activity.begin();
            assert!(activity.is_scanning());
        }
        // The inner scan ended; the outer one is still running.
        assert!(activity.is_scanning());
        drop(outer);
        assert!(!activity.is_scanning());
    }

    /// A scan that panics still has to release whoever is waiting on it.
    #[test]
    fn a_panicking_scan_releases_its_waiters() {
        let activity = ScanActivity::default();
        let scanner = activity.clone();
        let result = std::panic::catch_unwind(move || {
            let _scan = scanner.begin();
            panic!("scan blew up");
        });
        assert!(result.is_err());
        assert!(!activity.is_scanning());
    }
}
