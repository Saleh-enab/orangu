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
//! `orangu-server`'s own `-s`/`--shell-completions`. The positional
//! `MANIFEST` completes `.json` files, `-m`/`--model` `.gguf` files, and
//! `-q`/`--quantization` the weight formats — those by shelling back out to
//! `orangu-gguf --list-quantizations` itself and reading its first column,
//! the same trick `orangu-server`'s scripts use for models, so the list
//! never has to be repeated here. `-ts` and `-cs` — the two-letter short
//! options `main.rs`'s `normalize` rewrites — are offered alongside their
//! long forms. No clap-generated completion machinery is involved. The
//! PowerShell script is kept to ASCII: Windows PowerShell decodes a native
//! command's output in the console's code page, which would garble a
//! tooltip's dash.
//!
//! `every_flag_is_offered_by_every_completion_script` (`main.rs`) is what
//! keeps these three scripts in step with the parser.

pub const BASH: &str = r#"# bash completion for orangu-gguf
#
# Quick setup — add to ~/.bashrc:
#   eval "$(orangu-gguf -s)"
#
# Or write once to the bash-completion drop-in directory:
#   orangu-gguf -s > ~/.local/share/bash-completion/completions/orangu-gguf

# Completes -q/--quantization with every weight format from
# `orangu-gguf --list-quantizations`'s output.
_orangu_gguf_quantizations() {
    orangu-gguf --list-quantizations 2>/dev/null | awk '{print $1}'
}

_orangu_gguf() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()

    case "$prev" in
        -m|--model)
            COMPREPLY=( $(compgen -f -X '!*.gguf' -- "$cur") $(compgen -d -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        -o|--output|--flamegraph)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return 0
            ;;
        -q|--quantization)
            COMPREPLY=( $(compgen -W "$(_orangu_gguf_quantizations)" -- "$cur") )
            return 0
            ;;
        --flamegraph-call-graph)
            COMPREPLY=( $(compgen -W "fp dwarf" -- "$cur") )
            return 0
            ;;
        -ts|--training-size|-cs|--context-size|--flamegraph-freq)
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W \
            "-m --model -ts --training-size -q --quantization -cs --context-size -o --output --list-quantizations \
             --flamegraph --flamegraph-freq --flamegraph-call-graph --flamegraph-png -s --shell-completions -h --help -V --version" -- "$cur") )
        return 0
    fi

    COMPREPLY=( $(compgen -f -X '!*.json' -- "$cur") $(compgen -d -- "$cur") )
    compopt -o filenames 2>/dev/null
}

complete -F _orangu_gguf orangu-gguf
"#;

pub const ZSH: &str = r#"#compdef orangu-gguf
# zsh completion for orangu-gguf
#
# Quick setup — add to ~/.zshrc:
#   eval "$(orangu-gguf -s)"
#
# Or write once to your fpath directory:
#   orangu-gguf -s > ~/.zsh/completions/_orangu-gguf
#   # ~/.zshrc: fpath=(~/.zsh/completions $fpath) && autoload -Uz compinit && compinit

# Completes -q/--quantization with every weight format from
# `orangu-gguf --list-quantizations`'s output.
_orangu_gguf_quantizations() {
    local -a candidates
    candidates=( ${(f)"$(orangu-gguf --list-quantizations 2>/dev/null | awk '{print $1}')"} )
    compadd -a candidates
}

_orangu_gguf() {
    _arguments \
        '(-m --model)'{-m,--model}'[Convert this GGUF file instead of training a new model]:model:_files -g "*.gguf"' \
        '(-ts --training-size)'{-ts,--training-size}'[Training size for this run, overriding the manifest]:size:' \
        '(-q --quantization)'{-q,--quantization}'[Weight format written, overriding the manifest]:quantization:_orangu_gguf_quantizations' \
        '(-cs --context-size)'{-cs,--context-size}'[Context length the model declares, overriding the manifest]:n:' \
        '(-o --output)'{-o,--output}'[Where the model is written, overriding the manifest]:output:_files' \
        '--list-quantizations[List the weight formats a manifest'"'"'s quantization accepts]' \
        '--flamegraph[Record a CPU flamegraph of the run and render it here]:path:_files' \
        '--flamegraph-freq[Sampling frequency in Hz for --flamegraph]:hz:' \
        '--flamegraph-call-graph[Call-graph mode for --flamegraph]:mode:(fp dwarf)' \
        '--flamegraph-png[Also render a PNG beside the flamegraph SVG]' \
        '(-s --shell-completions)'{-s,--shell-completions}'[Print shell completion script for the detected shell and exit]' \
        '(-h --help)'{-h,--help}'[Print help]' \
        '(-V --version)'{-V,--version}'[Print version]' \
        '1:manifest:_files -g "*.json"'
}

_orangu_gguf "$@"
"#;

pub const FISH: &str = r#"# fish completion for orangu-gguf
#
# Quick setup — add to ~/.config/fish/config.fish:
#   orangu-gguf -s | source
#
# Or write once to the fish completions directory:
#   orangu-gguf -s > ~/.config/fish/completions/orangu-gguf.fish

# Completes -q/--quantization with every weight format from
# `orangu-gguf --list-quantizations`'s output.
function __orangu_gguf_quantizations
    orangu-gguf --list-quantizations 2>/dev/null | awk '{print $1}'
end

# The positional MANIFEST: a JSON file.
complete -c orangu-gguf -n '__fish_is_first_arg' -k -a '(__fish_complete_suffix .json)'
complete -c orangu-gguf -s m -l model              -r -k -a '(__fish_complete_suffix .gguf)' -d 'Convert this GGUF file instead of training a new model'
complete -c orangu-gguf -o ts -l training-size     -x -d 'Training size for this run, overriding the manifest'
complete -c orangu-gguf -s q -l quantization       -x -a '(__orangu_gguf_quantizations)' -d 'Weight format written, overriding the manifest'
complete -c orangu-gguf -o cs -l context-size      -x -d 'Context length the model declares, overriding the manifest'
complete -c orangu-gguf -s o -l output             -r -d 'Where the model is written, overriding the manifest'
complete -c orangu-gguf      -l list-quantizations    -d 'List the weight formats a manifest\'s quantization accepts'
complete -c orangu-gguf      -l flamegraph         -r -d 'Record a CPU flamegraph of the run and render it here'
complete -c orangu-gguf      -l flamegraph-freq    -x -d 'Sampling frequency in Hz for --flamegraph'
complete -c orangu-gguf      -l flamegraph-call-graph -x -a 'fp dwarf' -d 'Call-graph mode for --flamegraph'
complete -c orangu-gguf      -l flamegraph-png        -d 'Also render a PNG beside the flamegraph SVG'
complete -c orangu-gguf -s s -l shell-completions     -d 'Print shell completion script for the detected shell and exit'
complete -c orangu-gguf -s h -l help                  -d 'Print help'
complete -c orangu-gguf -s V -l version               -d 'Print version'
"#;

pub const POWERSHELL: &str = r#"# PowerShell completion for orangu-gguf
#
# Quick setup - add to your $PROFILE (notepad $PROFILE):
#   orangu-gguf -s | Out-String | Invoke-Expression
#
# Or write once to a file and dot-source it from $PROFILE:
#   orangu-gguf -s > "$HOME\orangu-gguf.ps1"
#   # $PROFILE: . "$HOME\orangu-gguf.ps1"

Register-ArgumentCompleter -Native -CommandName 'orangu-gguf' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    # Every word typed before the one under the cursor; $words[0] is the
    # command itself, so $prev is that when nothing else has been typed.
    $words = @($commandAst.CommandElements | ForEach-Object { $_.Extent.Text })
    if ($wordToComplete -ne '' -and $words.Count -gt 1) {
        $words = $words[0..($words.Count - 2)]
    }
    $prev = $words[-1]

    $flags = @(
        @('-m', '--model', 'Convert this GGUF file instead of training a new model'),
        @('-ts', '--training-size', 'Training size for this run, overriding the manifest'),
        @('-q', '--quantization', 'Weight format written, overriding the manifest'),
        @('-cs', '--context-size', 'Context length the model declares, overriding the manifest'),
        @('-o', '--output', 'Where the model is written, overriding the manifest'),
        @('--list-quantizations', 'List the weight formats a manifest''s quantization accepts'),
        @('--flamegraph', 'Record a CPU flamegraph of the run and render it here'),
        @('--flamegraph-freq', 'Sampling frequency in Hz for --flamegraph'),
        @('--flamegraph-call-graph', 'Call-graph mode for --flamegraph: fp or dwarf'),
        @('--flamegraph-png', 'Also render a PNG beside the flamegraph SVG'),
        @('-s', '--shell-completions', 'Print shell completion script for the detected shell and exit'),
        @('-h', '--help', 'Print help'),
        @('-V', '--version', 'Print version')
    )

    function Offer([string[]]$candidates) {
        foreach ($candidate in $candidates) {
            if ($candidate -like "$wordToComplete*") {
                [System.Management.Automation.CompletionResult]::new($candidate, $candidate, 'ParameterValue', $candidate)
            }
        }
    }

    # -q/--quantization: every weight format from `orangu-gguf --list-quantizations`.
    function Quantizations {
        & orangu-gguf --list-quantizations 2>$null | ForEach-Object { (-split $_)[0] }
    }

    switch ($prev) {
        # A path (the manifest, a model, an output): PowerShell's own completion takes over.
        { $_ -in '-m', '--model', '-o', '--output', '--flamegraph' } { return }
        { $_ -in '-q', '--quantization' } { return Offer (Quantizations) }
        '--flamegraph-call-graph' { return Offer @('fp', 'dwarf') }
        { $_ -in '-ts', '--training-size', '-cs', '--context-size', '--flamegraph-freq' } { return }
    }

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
