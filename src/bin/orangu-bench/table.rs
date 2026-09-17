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

//! `--table`: the history file as a markdown comparison.
//!
//! The chart answers "how does throughput behave as context grows"; a
//! document comparing two engines over twenty models needs the other view —
//! every model on its own row, one column per engine, and the ratio. That is
//! a table, and a table that goes into a markdown document should be written
//! by the tool that took the measurements rather than retyped from twenty
//! terminal transcripts, which is where transcription errors come from.
//!
//! The grouping comes from the label convention the sweep already writes:
//! `<label> · VAR=value`. Rows sharing a `<label>` are one model measured at
//! several values of the swept variable — `ENGINE=orangu` against
//! `ENGINE=llama.cpp`, say — so the prefix is the row and the value is the
//! column. Labels without that shape are each their own row, with the plain
//! label as the only column, so a file of ordinary runs still renders.
//!
//! Only the **latest** measurement of each point is shown, for the reason the
//! chart draws only the newest date: the file keeps every run, the table
//! answers what the build that exists does now.

use std::fmt::Write as _;

use super::history::Record;

/// One cell's provenance: which row of the file it came from, so the latest
/// wins and ties go to the faster repetition.
#[derive(Clone)]
struct Cell {
    date: String,
    best: f64,
    mean: f64,
    sd_sample: Option<f64>,
}

/// The modes a table shows: every one that is a rate or a cost per token.
const TABLED_MODES: &[&str] = &["pp", "tg", "pg", "curve", "cpu", "embed", "ttft"];

/// The `(row, column)` a record's label maps to.
///
/// `<prefix> · VAR=value` → `(prefix, Some(value))`; anything else is its own
/// row with no column value. The split is on the *last* ` · `, because a
/// `--label` may itself contain one (a device suffix, a build id).
fn split_label(label: &str) -> (String, Option<String>) {
    if let Some((prefix, point)) = label.rsplit_once(" · ")
        && let Some((_, value)) = point.split_once('=')
    {
        return (prefix.to_string(), Some(value.to_string()));
    }
    (label.to_string(), None)
}

/// Render `records` as markdown: one table per mode, in the order the modes
/// first appear, each with one row per model and context and one column per
/// swept value. With exactly two columns the last one is their ratio,
/// `first / second`, so with `ENGINE=orangu,llama.cpp` a ratio above 1 reads
/// as orangu ahead.
pub fn render(records: &[Record]) -> String {
    // Latest per (label, mode, n); at equal dates the better repetition.
    let mut latest: Vec<(String, String, u32, Cell)> = Vec::new();
    for r in records {
        let cell = Cell {
            date: r.date.clone(),
            best: r.best,
            mean: r.mean,
            sd_sample: r.sd_sample,
        };
        match latest
            .iter_mut()
            .find(|(l, m, n, _)| *l == r.label && *m == r.mode && *n == r.n)
        {
            Some((_, _, _, have)) => {
                if r.date > have.date || (r.date == have.date && r.best > have.best) {
                    *have = cell;
                }
            }
            None => latest.push((r.label.clone(), r.mode.clone(), r.n, cell)),
        }
    }

    // The throughput modes only. A run also records storage rows — bytes per
    // token, major faults — for the streaming regime, and those belong to the
    // chart's own panels; in a comparison table they would be a row of zeros
    // under every model that fits in RAM.
    latest.retain(|(_, mode, _, _)| TABLED_MODES.contains(&mode.as_str()));

    let mut modes: Vec<String> = Vec::new();
    let mut rows: Vec<String> = Vec::new();
    let mut columns: Vec<Option<String>> = Vec::new();
    for (label, mode, _, _) in &latest {
        if !modes.contains(mode) {
            modes.push(mode.clone());
        }
        let (row, column) = split_label(label);
        if !rows.contains(&row) {
            rows.push(row);
        }
        if !columns.contains(&column) {
            columns.push(column);
        }
    }
    // A file mixing swept and unswept labels has rows that never had a value;
    // they render under the plain-label column beside the swept ones, which
    // is the honest picture rather than an error.
    let ratio = columns.len() == 2 && columns.iter().all(Option::is_some);

    let column_name = |c: &Option<String>| c.clone().unwrap_or_else(|| "best".to_string());
    let mut out = String::new();
    for mode in &modes {
        let unit = match mode.as_str() {
            "cpu" => "ms/token, lower is better",
            "ttft" => "ms to first token, lower is better",
            "pg" => "tok/s, prompt + generated over the whole turn",
            _ => "tok/s",
        };
        let x = match mode.as_str() {
            "pp" | "embed" => "prompt",
            "tg" | "cpu" => "depth",
            "pg" => "prompt",
            _ => "n",
        };
        let _ = writeln!(out, "**{mode}** ({unit})\n");
        let _ = write!(out, "| model | {x} |");
        for c in &columns {
            let _ = write!(out, " {} |", column_name(c));
        }
        if ratio {
            let _ = write!(
                out,
                " {} / {} |",
                column_name(&columns[0]),
                column_name(&columns[1])
            );
        }
        let _ = writeln!(out);
        let _ = write!(out, "| :-- | --: |");
        for _ in &columns {
            let _ = write!(out, " --: |");
        }
        if ratio {
            let _ = write!(out, " --: |");
        }
        let _ = writeln!(out);

        for row in &rows {
            let mut ns: Vec<u32> = latest
                .iter()
                .filter(|(l, m, _, _)| m == mode && split_label(l).0 == *row)
                .map(|(_, _, n, _)| *n)
                .collect();
            ns.sort_unstable();
            ns.dedup();
            for n in ns {
                let at = |column: &Option<String>| -> Option<&Cell> {
                    latest
                        .iter()
                        .find(|(l, m, k, _)| {
                            m == mode && *k == n && {
                                let (r, c) = split_label(l);
                                r == *row && c == *column
                            }
                        })
                        .map(|(_, _, _, cell)| cell)
                };
                let _ = write!(out, "| {row} | {n} |");
                for c in &columns {
                    match at(c) {
                        Some(cell) => {
                            let _ = write!(out, " {} |", fmt_cell(cell));
                        }
                        None => {
                            let _ = write!(out, " — |");
                        }
                    }
                }
                if ratio {
                    match (at(&columns[0]), at(&columns[1])) {
                        (Some(a), Some(b)) if b.best > 0.0 => {
                            let _ = write!(out, " {:.2}× |", a.best / b.best);
                        }
                        _ => {
                            let _ = write!(out, " — |");
                        }
                    }
                }
                let _ = writeln!(out);
            }
        }
        let _ = writeln!(out);
    }
    out
}

/// `best` with the run's spread beside it where it has one: `71.5 (70.9 ± 0.4)`.
/// The headline is the best repetition, as everywhere else in this tool; the
/// mean and sample sd say how much to trust it.
fn fmt_cell(cell: &Cell) -> String {
    match cell.sd_sample {
        Some(sd) => format!("{:.1} ({:.1} ± {:.1})", cell.best, cell.mean, sd),
        None => format!("{:.1}", cell.best),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(date: &str, label: &str, mode: &str, n: u32, best: f64) -> Record {
        Record {
            date: date.to_string(),
            label: label.to_string(),
            mode: mode.to_string(),
            n,
            best,
            mean: best - 1.0,
            sd: 0.5,
            sd_sample: Some(0.6),
            device: None,
        }
    }

    #[test]
    fn a_sweep_label_splits_into_row_and_column() {
        assert_eq!(
            split_label("gemma-4-E2B · ENGINE=orangu"),
            ("gemma-4-E2B".to_string(), Some("orangu".to_string()))
        );
        // The last separator is the point; an earlier one belongs to the label.
        assert_eq!(
            split_label("orangu abc123 · card1 · ORANGU_DEVICE=1"),
            ("orangu abc123 · card1".to_string(), Some("1".to_string()))
        );
        assert_eq!(split_label("plain"), ("plain".to_string(), None));
        assert_eq!(
            split_label("a · b"),
            ("a · b".to_string(), None),
            "a suffix without `=` is not a sweep point"
        );
    }

    /// Two engines over two models: one row per model and context, the
    /// engines as columns, the ratio last, and the latest date winning.
    #[test]
    fn two_swept_values_render_as_columns_with_a_ratio() {
        let records = vec![
            rec("2026-09-01", "m1 · ENGINE=orangu", "tg", 0, 50.0),
            rec("2026-09-02", "m1 · ENGINE=orangu", "tg", 0, 60.0),
            rec("2026-09-02", "m1 · ENGINE=llama.cpp", "tg", 0, 80.0),
            rec("2026-09-02", "m2 · ENGINE=orangu", "tg", 0, 10.0),
            rec("2026-09-02", "m2 · ENGINE=llama.cpp", "tg", 0, 5.0),
            rec("2026-09-02", "m1 · ENGINE=orangu", "pp", 512, 400.0),
            rec("2026-09-02", "m1 · ENGINE=orangu", "io_majflt", 0, 12.0),
        ];
        let md = render(&records);
        assert!(md.contains("**tg** (tok/s)"), "{md}");
        assert!(
            md.contains("| model | depth | orangu | llama.cpp | orangu / llama.cpp |"),
            "{md}"
        );
        assert!(
            md.contains("| m1 | 0 | 60.0 (59.0 ± 0.6) | 80.0 (79.0 ± 0.6) | 0.75× |"),
            "{md}"
        );
        assert!(
            md.contains("| m2 | 0 | 10.0 (9.0 ± 0.6) | 5.0 (4.0 ± 0.6) | 2.00× |"),
            "{md}"
        );
        // The pp table has only one engine's row; the other cell is empty
        // and the ratio cannot be formed.
        assert!(
            md.contains("| m1 | 512 | 400.0 (399.0 ± 0.6) | — | — |"),
            "{md}"
        );
        assert!(
            !md.contains("io_majflt"),
            "storage rows are not throughput: {md}"
        );
    }

    /// Plain labels — an ordinary run's history — still render, as one row
    /// each with a single column and no ratio.
    #[test]
    fn plain_labels_render_one_row_each() {
        let records = vec![
            rec("2026-09-02", "orangu abc", "tg", 0, 60.0),
            rec("2026-09-02", "reference", "tg", 0, 80.0),
        ];
        let md = render(&records);
        assert!(md.contains("| model | depth | best |"), "{md}");
        assert!(
            md.contains("| orangu abc | 0 | 60.0 (59.0 ± 0.6) |\n"),
            "{md}"
        );
        assert!(
            md.contains("| reference | 0 | 80.0 (79.0 ± 0.6) |\n"),
            "{md}"
        );
        assert!(!md.contains('×'), "{md}");
    }
}
