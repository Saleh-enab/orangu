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

//! The `-s`/`--shell-completions` switch every orangu binary carries.
//!
//! Each binary keeps its own hand-written bash, zsh, fish and PowerShell
//! scripts (its `shell.rs`); what they share is the part that picks one of
//! the four from the environment, and the error that names the supported
//! shells — and the binary's own name — when it cannot. That part lives
//! here, once, so the five binaries cannot drift in what they accept or
//! how they refuse.

use anyhow::{Result, anyhow};

/// One binary's four completion scripts.
pub struct Scripts {
    pub bash: &'static str,
    pub zsh: &'static str,
    pub fish: &'static str,
    pub powershell: &'static str,
}

/// Pick the completion script for a `$SHELL` value. Separate from printing
/// it so the detection — including the shell it refuses — can be checked
/// without a process to read stdout from.
///
/// The value is matched by its last path component, with a `.exe` suffix
/// ignored, so `/bin/bash`, `bash`, `C:\Program Files\PowerShell\7\pwsh.exe`
/// and `powershell` all name what they look like. `in_powershell` is the
/// fallback for a `$SHELL` that names nothing: Windows sets no `$SHELL` at
/// all, and PowerShell — Windows PowerShell and pwsh alike — is the one
/// shell that announces itself another way, through `$PSModulePath`.
///
/// `binary` is the executable's own name (`orangu-server`, …): it is only
/// used in the error, whose usage lines show how to install the script for
/// each shell.
pub fn script(
    binary: &str,
    shell: &str,
    in_powershell: bool,
    scripts: &Scripts,
) -> Result<&'static str> {
    let name = shell
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let name = name.strip_suffix(".exe").unwrap_or(&name);
    match name {
        "bash" => Ok(scripts.bash),
        "zsh" => Ok(scripts.zsh),
        "fish" => Ok(scripts.fish),
        "pwsh" | "powershell" => Ok(scripts.powershell),
        _ if in_powershell => Ok(scripts.powershell),
        _ => Err(anyhow!(
            "could not detect shell from $SHELL ({shell:?}).\n\
             Supported shells: bash, zsh, fish, PowerShell.\n\
             \n\
             Usage:\n\
             \x20 bash:       eval \"$({binary} -s)\"\n\
             \x20 zsh:        {binary} -s > ~/.zsh/completions/_{binary}\n\
             \x20 fish:       {binary} -s > ~/.config/fish/completions/{binary}.fish\n\
             \x20 PowerShell: {binary} -s | Out-String | Invoke-Expression"
        )),
    }
}

/// Pick the script for the shell the environment names: `$SHELL`, or,
/// failing that, PowerShell when `$PSModulePath` says this is one.
pub fn detect(binary: &str, scripts: &Scripts) -> Result<&'static str> {
    script(
        binary,
        &std::env::var("SHELL").unwrap_or_default(),
        std::env::var_os("PSModulePath").is_some_and(|path| !path.is_empty()),
        scripts,
    )
}

/// Detect the shell and print its completion script to stdout.
pub fn print(binary: &str, scripts: &Scripts) -> Result<()> {
    print!("{}", detect(binary, scripts)?);
    Ok(())
}

/// Every flag `command` parses that one of the four scripts does not
/// offer, as `"<shell>: <flag>"` — empty when the scripts are in step with
/// the parser. Each binary's tests assert exactly that.
///
/// Exists because the scripts are hand-written, with no generator behind
/// them: a flag added to clap is silently missing from Tab until someone
/// notices. Hidden arguments are skipped, as they are from `--help`.
///
/// The check is lexical — a script offers `--foo` when `--foo` appears in
/// it as a word (bash, zsh and PowerShell), or as `-l foo` (fish) — which
/// is enough to catch the flag that was never added, and not a parser of
/// four shells' completion languages. `*` is a word break so zsh's `*--foo[...]`, the
/// spelling of a repeatable option, counts as offering `--foo`.
pub fn unoffered(command: &clap::Command, scripts: &Scripts) -> Vec<String> {
    let mut command = command.clone();
    command.build();
    let tokens = |script: &'static str| -> Vec<&'static str> {
        script
            .split(|c: char| c.is_whitespace() || "'\"{}[](),|=*".contains(c))
            .filter(|token| !token.is_empty())
            .collect()
    };
    let bash = tokens(scripts.bash);
    let zsh = tokens(scripts.zsh);
    let fish = tokens(scripts.fish);
    let powershell = tokens(scripts.powershell);
    let fish_offers = |option: &str, name: &str| {
        fish.windows(2)
            .any(|pair| pair[0] == option && pair[1] == name)
    };

    let mut missing = Vec::new();
    for arg in command.get_arguments().filter(|arg| !arg.is_hide_set()) {
        if let Some(long) = arg.get_long() {
            let flag = format!("--{long}");
            if !bash.contains(&flag.as_str()) {
                missing.push(format!("bash: {flag}"));
            }
            if !zsh.contains(&flag.as_str()) {
                missing.push(format!("zsh: {flag}"));
            }
            if !fish_offers("-l", long) {
                missing.push(format!("fish: {flag}"));
            }
            if !powershell.contains(&flag.as_str()) {
                missing.push(format!("powershell: {flag}"));
            }
        }
        if let Some(short) = arg.get_short() {
            let flag = format!("-{short}");
            if !bash.contains(&flag.as_str()) {
                missing.push(format!("bash: {flag}"));
            }
            if !zsh.contains(&flag.as_str()) {
                missing.push(format!("zsh: {flag}"));
            }
            if !fish_offers("-s", &short.to_string()) {
                missing.push(format!("fish: {flag}"));
            }
            if !powershell.contains(&flag.as_str()) {
                missing.push(format!("powershell: {flag}"));
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRIPTS: Scripts = Scripts {
        bash: "# bash completion",
        zsh: "#compdef",
        fish: "# fish completion",
        powershell: "# PowerShell completion",
    };

    #[test]
    fn detects_each_supported_shell_by_path_or_name() {
        for (shell, expected) in [
            ("/bin/bash", SCRIPTS.bash),
            ("bash", SCRIPTS.bash),
            ("/usr/bin/zsh", SCRIPTS.zsh),
            ("zsh", SCRIPTS.zsh),
            ("/usr/local/bin/fish", SCRIPTS.fish),
            ("fish", SCRIPTS.fish),
            ("pwsh", SCRIPTS.powershell),
            ("/usr/bin/pwsh", SCRIPTS.powershell),
            ("powershell", SCRIPTS.powershell),
            (
                r"C:\Program Files\PowerShell\7\pwsh.exe",
                SCRIPTS.powershell,
            ),
            (
                r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                SCRIPTS.powershell,
            ),
            // Git Bash on Windows.
            (r"C:\Program Files\Git\usr\bin\bash.exe", SCRIPTS.bash),
        ] {
            assert_eq!(
                script("orangu", shell, false, &SCRIPTS).expect(shell),
                expected
            );
        }
    }

    /// Windows sets no `$SHELL`; PowerShell there is known by
    /// `$PSModulePath` instead. But a `$SHELL` that does name a shell wins
    /// — Git Bash on Windows inherits `$PSModulePath` too.
    #[test]
    fn powershell_is_the_fallback_when_only_psmodulepath_says_so() {
        assert_eq!(
            script("orangu", "", true, &SCRIPTS).unwrap(),
            SCRIPTS.powershell
        );
        assert_eq!(
            script("orangu", "/bin/bash", true, &SCRIPTS).unwrap(),
            SCRIPTS.bash
        );
        assert!(script("orangu", "", false, &SCRIPTS).is_err());
    }

    #[test]
    fn refuses_an_undetectable_shell_naming_the_binary() {
        let err = script("orangu-bench", "/usr/bin/nonesuch", false, &SCRIPTS)
            .expect_err("an unsupported shell must not yield a script")
            .to_string();
        assert!(err.contains("nonesuch"), "{err}");
        assert!(err.contains("bash, zsh, fish, PowerShell"), "{err}");
        assert!(err.contains("eval \"$(orangu-bench -s)\""), "{err}");
        assert!(err.contains("_orangu-bench"), "{err}");
        assert!(err.contains("orangu-bench.fish"), "{err}");
        assert!(
            err.contains("orangu-bench -s | Out-String | Invoke-Expression"),
            "{err}"
        );
        assert!(script("orangu", "", false, &SCRIPTS).is_err());
    }

    /// The checker must be able to fail, or the tests it serves are
    /// decoration: a parser with one flag no script offers is reported by
    /// every shell, and a flag they all offer by none.
    #[test]
    fn unoffered_names_the_flag_every_script_lacks() {
        let command = clap::Command::new("tool")
            .arg(clap::Arg::new("known").short('k').long("known"))
            .arg(clap::Arg::new("added").short('a').long("added"))
            .arg(clap::Arg::new("repeat").long("repeat"))
            .arg(clap::Arg::new("secret").long("secret").hide(true));
        let scripts = Scripts {
            bash: "compgen -W \"-k --known --repeat -h --help -V --version\"",
            zsh: "'(-k --known)'{-k,--known}'[Known]' '*--repeat[Repeatable]' '(-h --help)'{-h,--help}'[Print help]' '(-V --version)'{-V,--version}'[Print version]'",
            fish: "complete -c tool -s k -l known\ncomplete -c tool -l repeat\ncomplete -c tool -s h -l help\ncomplete -c tool -s V -l version",
            powershell: "@('-k', '--known', 'Known'), @('--repeat', 'Repeatable'), @('-h', '--help', 'Print help'), @('-V', '--version', 'Print version')",
        };
        assert_eq!(
            unoffered(&command, &scripts),
            [
                "bash: --added",
                "zsh: --added",
                "fish: --added",
                "powershell: --added",
                "bash: -a",
                "zsh: -a",
                "fish: -a",
                "powershell: -a"
            ]
        );
    }
}
