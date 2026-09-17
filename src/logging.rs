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

//! Where `orangu-server` and `orangu-coordinator` write what they have to
//! say while they run — the terminal, or a file — resolved from the same two
//! keys in either binary's config section: `log_type` (`console`, the
//! default, or `file`) and `log_path` (for `file`: the file, defaulting to
//! `orangu-server.log`/`orangu-coordinator.log` in the directory the process
//! was started from).
//!
//! Built on the `log` facade with `fern` behind it, and the console output
//! is exactly what it was before either existed: a record is the bare
//! message, `info` on stdout and `warn`/`error` on stderr, which is where
//! `println!`/`eprintln!` put the same lines. Only a file gets a timestamp
//! and a level in front of each line — a line in a file is read a week
//! later, when "which line came first" is no longer obvious.
//!
//! A file is opened for appending and shared: `orangu-coordinator` forwards
//! its own `log_type`/`log_path` into the config it generates for every
//! profile's `orangu-server`, so the two write into one file, each line
//! stamped by whichever process wrote it. That is also why a file is where
//! the once-a-second progress a request prints to a terminal is *not*
//! written — a file gets each request's completed line, not its cursor
//! movements.
//!
//! What is logged is the *serving* output: the startup banner, every
//! request's completion line, and the notes and warnings a running server
//! produces. A subcommand's own output (`list`, `show`, `system`, ...) and
//! an interactive prompt are terminal conversations, not log lines, and stay
//! on plain `println!` — so do the fatal `error: ...` lines a process exits
//! on, which belong on the terminal that started it whatever the config
//! says about a log file.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

/// The `log_type` value that keeps the console — the default, and what
/// `--init` offers first.
pub const LOG_TYPE_CONSOLE: &str = "console";
/// The `log_type` value that sends everything to `log_path` instead.
pub const LOG_TYPE_FILE: &str = "file";
/// Every accepted `log_type`, in the order a prompt offers them.
pub const LOG_TYPES: [&str; 2] = [LOG_TYPE_CONSOLE, LOG_TYPE_FILE];

/// Where a process logs — `log_type`/`log_path`, resolved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LogTarget {
    /// `log_type = console` (or no `log_type` at all): the terminal, exactly
    /// as before the key existed.
    #[default]
    Console,
    /// `log_type = file`: everything is appended to this file — `log_path`,
    /// or [`default_log_path`] when that says nothing. Absolute — a daemon
    /// changes its working directory to `/` once detached, so a relative
    /// path is resolved against the directory the process was started in
    /// *before* that happens, and a leading `~` against the home directory,
    /// same as a `models` key.
    File(PathBuf),
}

/// The file `log_type = file` writes to when `log_path` says nothing:
/// `<section>.log` — `orangu-server.log` or `orangu-coordinator.log`, each
/// config section being named after its binary — in the directory the
/// process was started from, resolved right now so that a daemon's later
/// move to `/` doesn't change where the file is.
pub fn default_log_path(section: &str) -> PathBuf {
    absolute(PathBuf::from(format!("{section}.log")))
}

impl LogTarget {
    /// Resolves a section's `log_type` and `log_path` keys, as read from the
    /// config file (`None` for an absent key; a present-but-blank value is
    /// treated the same, matching every other optional key those files
    /// have).
    ///
    /// `log_path` is only consulted when `log_type = file` — a path with
    /// nothing to send to it is not an error, so a config can keep one
    /// written down while switched back to the console — and falls back to
    /// [`default_log_path`] when absent. An unknown `log_type` is an error
    /// naming the two accepted values rather than a silent console, since
    /// the whole point of setting it is to have the output somewhere else.
    pub fn from_keys(
        section: &str,
        log_type: Option<&str>,
        log_path: Option<&str>,
    ) -> Result<Self> {
        let log_type = log_type
            .map(|value| value.trim().to_lowercase())
            .filter(|value| !value.is_empty());
        match log_type.as_deref() {
            None | Some(LOG_TYPE_CONSOLE) => Ok(LogTarget::Console),
            Some(LOG_TYPE_FILE) => Ok(LogTarget::File(
                match log_path.map(str::trim).filter(|value| !value.is_empty()) {
                    Some(path) => absolute(expand_tilde(path)),
                    None => default_log_path(section),
                },
            )),
            Some(other) => Err(anyhow!(
                "invalid value for [{section}].log_type: '{other}' (expected {})",
                LOG_TYPES.join(" or ")
            )),
        }
    }

    /// The `log_type` spelling this target answers to.
    pub fn log_type(&self) -> &'static str {
        match self {
            LogTarget::Console => LOG_TYPE_CONSOLE,
            LogTarget::File(_) => LOG_TYPE_FILE,
        }
    }

    /// The file, when there is one.
    pub fn path(&self) -> Option<&Path> {
        match self {
            LogTarget::Console => None,
            LogTarget::File(path) => Some(path),
        }
    }

    pub fn is_console(&self) -> bool {
        matches!(self, LogTarget::Console)
    }
}

/// How much of the log reaches the **console**. Says nothing about a file,
/// which always gets everything: a daemon logging to a file logs exactly
/// what an attached run would, and there is nothing on a terminal for
/// `--quiet` to be quiet about once the log is a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Console {
    /// Everything — an attached run.
    Everything,
    /// Errors only — `--quiet`, which keeps the reason a process exits.
    ErrorsOnly,
    /// Nothing at all — `--daemon` with the console as its log. Detached,
    /// its stdout and stderr are `/dev/null`; there is no logging, and
    /// saying so here is clearer than writing into `/dev/null`.
    Nothing,
}

/// Installs this process's logger, once. `crate_name` is the binary's own
/// crate (`orangu_server`, `orangu_coordinator`): records from it and from
/// this library are logged at `info` and above, and records from every
/// dependency are dropped — `wgpu` and `hyper` alone would otherwise say
/// more than the server does.
///
/// A file is created if it doesn't exist (its directory too), and appended
/// to if it does. Opening it is the one thing here that can fail, and it
/// fails *here* — before a daemon detaches — so an unwritable `log_path` is
/// reported on the terminal with a non-zero exit, not discovered as a file
/// that never appears.
pub fn install(target: &LogTarget, console: Console, crate_name: &'static str) -> Result<()> {
    let level = match (target, console) {
        (LogTarget::File(_), _) | (LogTarget::Console, Console::Everything) => {
            log::LevelFilter::Info
        }
        (LogTarget::Console, Console::ErrorsOnly) => log::LevelFilter::Error,
        (LogTarget::Console, Console::Nothing) => log::LevelFilter::Off,
    };
    let dispatch = fern::Dispatch::new()
        .level(log::LevelFilter::Off)
        .level_for("orangu", level)
        .level_for(crate_name, level);

    let dispatch = match target {
        LogTarget::Console => dispatch
            .chain(
                fern::Dispatch::new()
                    .filter(|metadata| metadata.level() == log::Level::Info)
                    .format(|out, message, _record| out.finish(format_args!("{message}")))
                    .chain(std::io::stdout()),
            )
            .chain(
                fern::Dispatch::new()
                    .filter(|metadata| metadata.level() <= log::Level::Warn)
                    .format(|out, message, _record| out.finish(format_args!("{message}")))
                    .chain(std::io::stderr()),
            ),
        LogTarget::File(path) => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create log directory {}", parent.display())
                })?;
            }
            let file = fern::log_file(path)
                .with_context(|| format!("failed to open log file {}", path.display()))?;
            dispatch
                .format(|out, message, record| {
                    out.finish(format_args!(
                        "{} {:<5} {message}",
                        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                        record.level()
                    ))
                })
                .chain(file)
        }
    };
    dispatch
        .apply()
        .map_err(|_| anyhow!("a logger is already installed in this process"))
}

/// Expands a leading `~` or `~/` to the home directory — the same shorthand
/// both binaries' `models` keys accept.
fn expand_tilde(path: &str) -> PathBuf {
    match path.strip_prefix('~') {
        Some(rest) => match home::home_dir() {
            Some(home) => home.join(rest.trim_start_matches('/')),
            None => PathBuf::from(path),
        },
        None => PathBuf::from(path),
    }
}

/// Resolves a relative path against the current directory, without
/// touching the filesystem — the file need not exist yet. Falls back to the
/// path as given if the current directory can't be read at all.
fn absolute(path: PathBuf) -> PathBuf {
    std::path::absolute(&path).unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No key, a blank key, and `console` in any casing are all the console.
    #[test]
    fn the_console_is_the_default_and_needs_no_path() {
        for log_type in [None, Some(""), Some("  "), Some("console"), Some("CONSOLE")] {
            assert_eq!(
                LogTarget::from_keys("orangu-server", log_type, None).unwrap(),
                LogTarget::Console,
                "log_type = {log_type:?}"
            );
        }
        // A path with the console selected is kept but not used.
        assert_eq!(
            LogTarget::from_keys("orangu-server", None, Some("/var/log/orangu.log")).unwrap(),
            LogTarget::Console
        );
    }

    #[test]
    fn a_file_needs_a_path_and_gets_an_absolute_one() {
        // An absolute path is kept as given — spelled from the current
        // directory so it is absolute on every platform (a bare `/var/...`
        // has no drive on Windows and would get one prepended).
        let given = std::env::current_dir()
            .unwrap()
            .join("var")
            .join("orangu.log");
        let target = LogTarget::from_keys("orangu-server", Some("file"), given.to_str()).unwrap();
        assert_eq!(target, LogTarget::File(given.clone()));
        assert_eq!(target.log_type(), "file");
        assert_eq!(target.path(), Some(given.as_path()));

        // A relative path is anchored where the process started, since a
        // daemon moves to `/` before it writes a line.
        let target =
            LogTarget::from_keys("orangu-server", Some("FILE"), Some("orangu.log")).unwrap();
        let LogTarget::File(path) = target else {
            panic!("expected a file");
        };
        assert!(path.is_absolute(), "{}", path.display());
        assert!(path.ends_with("orangu.log"), "{}", path.display());
    }

    #[test]
    fn expands_a_leading_tilde_in_the_path() {
        let target =
            LogTarget::from_keys("orangu-coordinator", Some("file"), Some("~/orangu.log")).unwrap();
        assert_eq!(
            target.path(),
            Some(home::home_dir().unwrap().join("orangu.log").as_path())
        );
    }

    /// `file` with no path — absent or blank alike — is the section's own
    /// name with `.log`, in the directory the process was started from.
    #[test]
    fn a_file_without_a_path_defaults_beside_the_process() {
        let expected = std::env::current_dir()
            .unwrap()
            .join("orangu-coordinator.log");
        assert_eq!(default_log_path("orangu-coordinator"), expected);
        for log_path in [None, Some(""), Some("  ")] {
            assert_eq!(
                LogTarget::from_keys("orangu-coordinator", Some("file"), log_path).unwrap(),
                LogTarget::File(expected.clone()),
                "log_path = {log_path:?}"
            );
        }
        assert_eq!(
            default_log_path("orangu-server"),
            std::env::current_dir().unwrap().join("orangu-server.log")
        );
    }

    #[test]
    fn an_unknown_log_type_is_rejected_with_the_alternatives() {
        let err = LogTarget::from_keys("orangu-server", Some("syslog"), None).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("[orangu-server].log_type"), "{text}");
        assert!(text.contains("syslog"), "{text}");
        assert!(text.contains("console or file"), "{text}");
    }
}
