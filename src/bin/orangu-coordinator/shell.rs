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

//! Hand-written shell completion scripts, mirroring `orangu`'s and
//! `orangu-server`'s own `-s`/`--shell-completions`. The coordinator takes
//! no positional argument and no subcommand, so there is nothing to shell
//! back out for: only the flags are offered, with `-c`/`--config` completing
//! files. No clap-generated completion machinery is involved. The
//! PowerShell script is kept to ASCII: Windows PowerShell decodes a native
//! command's output in the console's code page, which would garble a
//! tooltip's dash.
//!
//! `every_flag_is_offered_by_every_completion_script` (`main.rs`) is what
//! keeps these three scripts in step with the parser.

pub const BASH: &str = r#"# bash completion for orangu-coordinator
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-coordinator -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-coordinator -s > ~/.local/share/bash-completion/completions/orangu-coordinator

_orangu_coordinator() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    case "$prev" in
        -c|--config)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "-c --config -i --init -q --quiet -d --daemon -s --shell-completions -h --help -V --version" -- "$cur") )
        return 0
    fi
}

complete -F _orangu_coordinator orangu-coordinator
"#;

pub const ZSH: &str = r#"#compdef orangu-coordinator
# zsh completion for orangu-coordinator
#
# Quick setup — add to ~/.zshrc:
#   eval "$(orangu-coordinator -s)"
#
# Or write once to your fpath directory:
#   orangu-coordinator -s > ~/.zsh/completions/_orangu-coordinator
#   # ~/.zshrc: fpath=(~/.zsh/completions $fpath) && autoload -Uz compinit && compinit

_orangu_coordinator() {
    _arguments \
        '(-c --config)'{-c,--config}'[Path to orangu-coordinator.conf (default: ./orangu-coordinator.conf, then ~/.orangu/orangu-coordinator.conf)]:config file:_files' \
        '(-i --init)'{-i,--init}'[Interactively create ~/.orangu/orangu-coordinator.conf and exit]' \
        '(-q --quiet)'{-q,--quiet}'[Suppress all output (the startup banner, profile list, and shutdown message)]' \
        '(-d --daemon)'{-d,--daemon}'[Detach from the terminal and run in the background (implies --quiet)]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]'
}

_orangu_coordinator "$@"
"#;

pub const FISH: &str = r#"# fish completion for orangu-coordinator
#
# Quick setup — add to ~/.config/fish/config.fish:
#   orangu-coordinator -s | source
#
# Or write once to the fish completions directory:
#   orangu-coordinator -s > ~/.config/fish/completions/orangu-coordinator.fish

complete -c orangu-coordinator -f
complete -c orangu-coordinator -s c -l config            -r -d 'Path to orangu-coordinator.conf (default: ./orangu-coordinator.conf, then ~/.orangu/orangu-coordinator.conf)'
complete -c orangu-coordinator -s i -l init                 -d 'Interactively create ~/.orangu/orangu-coordinator.conf and exit'
complete -c orangu-coordinator -s q -l quiet                -d 'Suppress all output (the startup banner, profile list, and shutdown message)'
complete -c orangu-coordinator -s d -l daemon               -d 'Detach from the terminal and run in the background (implies --quiet)'
complete -c orangu-coordinator -s s -l shell-completions    -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-coordinator -s h -l help                 -d 'Print help'
complete -c orangu-coordinator -s V -l version              -d 'Print version'
"#;

pub const POWERSHELL: &str = r#"# PowerShell completion for orangu-coordinator
#
# Quick setup - add to your $PROFILE (notepad $PROFILE):
#   orangu-coordinator -s | Out-String | Invoke-Expression
#
# Or write once to a file and dot-source it from $PROFILE:
#   orangu-coordinator -s > "$HOME\orangu-coordinator.ps1"
#   # $PROFILE: . "$HOME\orangu-coordinator.ps1"

Register-ArgumentCompleter -Native -CommandName 'orangu-coordinator' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    # Every word typed before the one under the cursor; $words[0] is the
    # command itself, so $prev is that when nothing else has been typed.
    $words = @($commandAst.CommandElements | ForEach-Object { $_.Extent.Text })
    if ($wordToComplete -ne '' -and $words.Count -gt 1) {
        $words = $words[0..($words.Count - 2)]
    }
    $prev = $words[-1]

    $flags = @(
        @('-c', '--config', 'Path to orangu-coordinator.conf (default: ./orangu-coordinator.conf, then ~/.orangu/orangu-coordinator.conf)'),
        @('-i', '--init', 'Interactively create ~/.orangu/orangu-coordinator.conf and exit'),
        @('-q', '--quiet', 'Suppress all output (the startup banner, profile list, and shutdown message)'),
        @('-d', '--daemon', 'Detach from the terminal and run in the background (implies --quiet)'),
        @('-s', '--shell-completions', 'Print shell completion script for the detected shell and exit'),
        @('-h', '--help', 'Print help'),
        @('-V', '--version', 'Print version')
    )

    if ($prev -in '-c', '--config') { return }   # a file: PowerShell's own completion takes over

    if ($wordToComplete.StartsWith('-')) {
        foreach ($flag in $flags) {
            $tooltip = $flag[-1]
            foreach ($name in $flag[0..($flag.Count - 2)]) {
                if ($name -like "$wordToComplete*") {
                    [System.Management.Automation.CompletionResult]::new($name, $name, 'ParameterName', $tooltip)
                }
            }
        }
    }
}
"#;
