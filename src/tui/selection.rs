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

//! Mouse text selection over the drawn screen.
//!
//! With mouse capture on, the terminal no longer selects text itself, so the
//! TUI does what the terminal would have done: a left-button drag highlights
//! the cells between the press and the pointer, and the release copies their
//! text. The selection is made of screen cells, not transcript lines — it
//! follows whatever the last frame put on screen, so it works the same in the
//! main window, `/review`, `/auto_review`, and `/manual`, and copies exactly
//! what is visible (wrapped, panned, or clipped as drawn).
//!
//! The screen is one resource, so its selection is one value: the live state
//! lives behind a process-wide lock rather than being threaded through every
//! event loop. Each frame hands its finished buffer to [`finish_frame`], which
//! paints the highlight and keeps a copy of the cells for the text to be read
//! from when the button is released. The pure functions ([`highlight`],
//! [`text`], [`word_at`]) work on any buffer and carry the tests.

use ratatui::buffer::Buffer;
use ratatui::style::Modifier;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthStr;

/// How long the `Copied …` notice stays on the status line.
pub const NOTICE_DURATION: Duration = Duration::from_secs(3);

/// A screen cell as `(column, row)`.
pub type Cell = (u16, u16);

/// A stream selection: every cell from `anchor` to `head` in reading order,
/// both ends included, whole rows in between. Either end may be the one the
/// button went down on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    anchor: Cell,
    head: Cell,
    /// The button is still down: releases finish the selection, and the
    /// pointer keeps moving `head`.
    dragging: bool,
}

impl Selection {
    /// Start a selection at the cell the button went down on.
    pub fn begin(cell: Cell) -> Self {
        Self {
            anchor: cell,
            head: cell,
            dragging: true,
        }
    }

    /// A finished selection covering `start..=end` in reading order.
    pub fn span(start: Cell, end: Cell) -> Self {
        Self {
            anchor: start,
            head: end,
            dragging: false,
        }
    }

    /// Move the free end to where the pointer is now.
    pub fn extend(&mut self, cell: Cell) {
        self.head = cell;
    }

    /// The button came up; the selection keeps its extent.
    pub fn finish(&mut self) {
        self.dragging = false;
    }

    pub fn is_dragging(&self) -> bool {
        self.dragging
    }

    /// A click without a drag selects nothing.
    pub fn is_empty(&self) -> bool {
        self.anchor == self.head
    }

    /// The two ends in reading order: `(first, last)`, both inclusive.
    pub fn bounds(&self) -> (Cell, Cell) {
        let a = (self.anchor.1, self.anchor.0);
        let h = (self.head.1, self.head.0);
        if a <= h {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }

    /// Whether `(column, row)` is inside the selection.
    pub fn contains(&self, cell: Cell) -> bool {
        let ((first_col, first_row), (last_col, last_row)) = self.bounds();
        let (col, row) = cell;
        if row < first_row || row > last_row {
            return false;
        }
        (row > first_row || col >= first_col) && (row < last_row || col <= last_col)
    }

    /// The columns of `row` inside the selection, clipped to `width`.
    fn columns(&self, row: u16, width: u16) -> Option<std::ops::RangeInclusive<u16>> {
        let ((first_col, first_row), (last_col, last_row)) = self.bounds();
        if row < first_row || row > last_row || width == 0 {
            return None;
        }
        let from = if row == first_row { first_col } else { 0 };
        let to = if row == last_row { last_col } else { width - 1 };
        let to = to.min(width - 1);
        (from <= to).then_some(from..=to)
    }
}

/// Paint the selection onto `buffer` in reverse video. Reverse video reads
/// against every theme and every background the screen already uses (the user
/// input band, code blocks, the header), which a fixed colour would not.
pub fn highlight(buffer: &mut Buffer, selection: &Selection) {
    let area = buffer.area;
    for row in area.top()..area.bottom() {
        let Some(columns) = selection.columns(row, area.width) else {
            continue;
        };
        for col in columns {
            if let Some(cell) = buffer.cell_mut((col, row)) {
                cell.modifier.insert(Modifier::REVERSED);
            }
        }
    }
}

/// Whether the cell at `(col, row)` is the hidden half of a wide character:
/// the buffer keeps the character in the first cell and a blank in the second,
/// which the terminal never shows.
fn hidden_by_wide(buffer: &Buffer, col: u16, row: u16) -> bool {
    col > buffer.area.left()
        && buffer
            .cell((col - 1, row))
            .is_some_and(|first| UnicodeWidthStr::width(first.symbol()) > 1)
}

/// The text under the selection: one line per screen row, trailing blanks
/// trimmed, rows joined with `\n`. A wide character takes two cells but
/// contributes its symbol once.
pub fn text(buffer: &Buffer, selection: &Selection) -> String {
    let area = buffer.area;
    let mut lines = Vec::new();
    for row in area.top()..area.bottom() {
        let Some(columns) = selection.columns(row, area.width) else {
            continue;
        };
        let line: String = columns
            .filter(|&col| !hidden_by_wide(buffer, col, row))
            .filter_map(|col| buffer.cell((col, row)))
            .map(|cell| cell.symbol())
            .collect();
        lines.push(line.trim_end().to_string());
    }
    lines.join("\n")
}

/// Whether `symbol` is part of a word for double-click selection. Besides
/// letters and digits this keeps the characters paths, identifiers, and
/// version numbers are made of, so a double-click on `src/tui/screen.rs` or
/// `snake_case-name` takes the whole thing.
fn is_word_symbol(symbol: &str) -> bool {
    symbol
        .chars()
        .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | '~' | '@'))
        && !symbol.is_empty()
}

/// The word under `(column, row)` as a finished selection, or `None` when the
/// cell is blank or outside the screen.
pub fn word_at(buffer: &Buffer, cell: Cell) -> Option<Selection> {
    let (col, row) = cell;
    let area = buffer.area;
    if row < area.top() || row >= area.bottom() || col < area.left() || col >= area.right() {
        return None;
    }
    // The hidden half of a wide character belongs to the word iff the
    // character does.
    let is_word = |c: u16| -> bool {
        let c = if hidden_by_wide(buffer, c, row) {
            c - 1
        } else {
            c
        };
        buffer
            .cell((c, row))
            .is_some_and(|cell| is_word_symbol(cell.symbol()))
    };
    if !is_word(col) {
        return None;
    }
    let mut first = col;
    while first > area.left() && is_word(first - 1) {
        first -= 1;
    }
    let mut last = col;
    while last + 1 < area.right() && is_word(last + 1) {
        last += 1;
    }
    Some(Selection::span((first, row), (last, row)))
}

/// The live selection, the frame it was made on, and the copy notice.
#[derive(Default)]
struct State {
    selection: Option<Selection>,
    /// The cells of the last frame drawn, read when the button comes up.
    frame: Option<Buffer>,
    notice: Option<(String, Instant)>,
}

static STATE: Mutex<State> = Mutex::new(State {
    selection: None,
    frame: None,
    notice: None,
});

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    // A poisoned lock only means a panic elsewhere while it was held; the
    // selection is cosmetic, so keep going with whatever is in it.
    let mut state = STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut state)
}

/// Called with every finished frame: paints the current selection over it and
/// keeps its cells for [`release`].
pub fn finish_frame(buffer: &mut Buffer) {
    with_state(|state| {
        state.frame = Some(buffer.clone());
        if let Some(selection) = &state.selection {
            highlight(buffer, selection);
        }
    });
}

/// The left button went down on `cell`: any earlier selection is dropped and a
/// new one starts here. Returns whether the screen changed.
pub fn press(cell: Cell) -> bool {
    with_state(|state| {
        let had_selection = state.selection.is_some();
        state.selection = Some(Selection::begin(cell));
        state.notice = None;
        had_selection
    })
}

/// The pointer moved to `cell` with the button down. Returns whether the
/// selection changed.
pub fn drag(cell: Cell) -> bool {
    with_state(|state| match &mut state.selection {
        Some(selection) if selection.is_dragging() && selection.head != cell => {
            selection.extend(cell);
            true
        }
        _ => false,
    })
}

/// The left button came up. A drag over at least two cells yields the selected
/// text, which stays highlighted; a plain click yields nothing and clears the
/// highlight. Only a selection still being dragged is finished here, so the
/// release that ends a double-click leaves its word selection alone.
pub fn release() -> Option<String> {
    with_state(|state| {
        let selection = state.selection.as_mut().filter(|s| s.is_dragging())?;
        selection.finish();
        if selection.is_empty() {
            state.selection = None;
            return None;
        }
        let selection = *selection;
        state.frame.as_ref().map(|frame| text(frame, &selection))
    })
}

/// Select the word under `cell` on the last frame drawn (a double-click) and
/// return its text; `None` when there is no word there.
pub fn select_word(cell: Cell) -> Option<String> {
    with_state(|state| {
        let frame = state.frame.as_ref()?;
        let selection = word_at(frame, cell)?;
        let word = text(frame, &selection);
        state.selection = Some(selection);
        Some(word)
    })
}

/// Drop the selection and its highlight. Returns whether there was one.
pub fn clear() -> bool {
    with_state(|state| state.selection.take().is_some())
}

/// Whether a selection is on screen.
pub fn is_active() -> bool {
    with_state(|state| state.selection.is_some())
}

/// Put `message` on the status line for [`NOTICE_DURATION`].
pub fn set_notice(message: String) {
    with_state(|state| state.notice = Some((message, Instant::now() + NOTICE_DURATION)));
}

/// The status-line notice, while it lasts.
pub fn notice() -> Option<String> {
    with_state(|state| {
        let (message, until) = state.notice.as_ref()?;
        if Instant::now() < *until {
            Some(message.clone())
        } else {
            state.notice = None;
            None
        }
    })
}

/// How long the notice has left, for event loops that poll with a timeout and
/// need to redraw when it expires.
pub fn notice_remaining() -> Option<Duration> {
    with_state(|state| {
        let (_, until) = state.notice.as_ref()?;
        Some(until.saturating_duration_since(Instant::now()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::layout::Rect;

    fn buffer(rows: &[&str]) -> Buffer {
        let width = rows
            .iter()
            .map(|r| UnicodeWidthStr::width(*r))
            .max()
            .unwrap_or(0) as u16;
        let mut buffer = Buffer::empty(Rect::new(0, 0, width, rows.len() as u16));
        for (y, row) in rows.iter().enumerate() {
            buffer.set_string(0, y as u16, row, ratatui::style::Style::default());
        }
        buffer
    }

    #[test]
    fn bounds_are_in_reading_order_whichever_end_is_the_anchor() {
        let mut forward = Selection::begin((5, 1));
        forward.extend((2, 3));
        let mut backward = Selection::begin((2, 3));
        backward.extend((5, 1));
        assert_eq!(forward.bounds(), ((5, 1), (2, 3)));
        assert_eq!(backward.bounds(), ((5, 1), (2, 3)));
    }

    #[test]
    fn contains_follows_the_stream_not_the_rectangle() {
        let selection = Selection::span((5, 1), (2, 3));
        assert!(selection.contains((5, 1)));
        assert!(selection.contains((79, 1)));
        assert!(selection.contains((0, 2)));
        assert!(selection.contains((2, 3)));
        assert!(!selection.contains((4, 1)));
        assert!(!selection.contains((3, 3)));
        assert!(!selection.contains((0, 0)));
        assert!(!selection.contains((0, 4)));
    }

    #[test]
    fn text_takes_whole_rows_between_the_ends_and_trims_each_row() {
        let buffer = buffer(&["first row  ", "second     ", "third row  "]);
        let selection = Selection::span((6, 0), (4, 2));
        assert_eq!(text(&buffer, &selection), "row\nsecond\nthird");
    }

    #[test]
    fn text_on_one_row_is_the_cells_between_the_ends() {
        let buffer = buffer(&["hello world"]);
        let selection = Selection::span((2, 0), (6, 0));
        assert_eq!(text(&buffer, &selection), "llo w");
    }

    #[test]
    fn text_reassembles_wide_characters() {
        let buffer = buffer(&["a日本b"]);
        let selection = Selection::span((0, 0), (5, 0));
        assert_eq!(text(&buffer, &selection), "a日本b");
    }

    #[test]
    fn text_clips_to_the_buffer() {
        let buffer = buffer(&["abc", "def"]);
        let selection = Selection::span((1, 0), (40, 7));
        assert_eq!(text(&buffer, &selection), "bc\ndef");
    }

    #[test]
    fn highlight_reverses_exactly_the_selected_cells() {
        let mut buffer = buffer(&["abcd", "efgh"]);
        highlight(&mut buffer, &Selection::span((2, 0), (0, 1)));
        let reversed = |x, y| {
            buffer
                .cell((x, y))
                .unwrap()
                .modifier
                .contains(Modifier::REVERSED)
        };
        assert!(!reversed(1, 0));
        assert!(reversed(2, 0));
        assert!(reversed(3, 0));
        assert!(reversed(0, 1));
        assert!(!reversed(1, 1));
    }

    #[test]
    fn word_at_takes_paths_and_identifiers_whole() {
        let buffer = buffer(&["see src/tui/screen.rs:12 (snake_case-name)"]);
        let word = |col| word_at(&buffer, (col, 0)).map(|s| text(&buffer, &s));
        assert_eq!(word(0).as_deref(), Some("see"));
        assert_eq!(word(9).as_deref(), Some("src/tui/screen.rs"));
        assert_eq!(word(21).as_deref(), None);
        assert_eq!(word(23).as_deref(), Some("12"));
        assert_eq!(word(25).as_deref(), None);
        assert_eq!(word(30).as_deref(), Some("snake_case-name"));
    }

    #[test]
    fn word_at_outside_the_screen_or_on_a_blank_is_nothing() {
        let buffer = buffer(&["ab cd"]);
        assert_eq!(word_at(&buffer, (2, 0)), None);
        assert_eq!(word_at(&buffer, (9, 0)), None);
        assert_eq!(word_at(&buffer, (0, 3)), None);
    }

    #[test]
    fn word_at_includes_wide_characters_from_either_cell() {
        let buffer = buffer(&["x日本y z"]);
        assert_eq!(
            word_at(&buffer, (2, 0))
                .map(|s| text(&buffer, &s))
                .as_deref(),
            Some("x日本y")
        );
        assert_eq!(
            word_at(&buffer, (5, 0))
                .map(|s| text(&buffer, &s))
                .as_deref(),
            Some("x日本y")
        );
    }

    #[test]
    fn a_click_without_a_drag_is_empty() {
        let mut selection = Selection::begin((3, 3));
        assert!(selection.is_empty());
        selection.extend((3, 3));
        assert!(selection.is_empty());
        selection.extend((4, 3));
        assert!(!selection.is_empty());
    }
}
