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

//! Shared human-readable size formatting, used wherever a byte count (a
//! model file's size, RAM/VRAM figures, ...) is shown to a user.

/// Formats a byte count as a human-readable size (e.g. `4.92 GiB`).
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

/// A duration as a person would say it — `40 s`, `6 min`, `1 h 20 min` —
/// for a wait that is an estimate, so no more digits than the estimate
/// deserves.
pub fn format_duration_rough(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    if seconds < 60 {
        return format!("{seconds} s");
    }
    let minutes = seconds.div_ceil(60);
    if minutes < 60 {
        return format!("{minutes} min");
    }
    let (hours, minutes) = (minutes / 60, minutes % 60);
    if minutes == 0 {
        format!("{hours} h")
    } else {
        format!("{hours} h {minutes} min")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rough_durations_read_like_speech() {
        assert_eq!(format_duration_rough(0.0), "0 s");
        assert_eq!(format_duration_rough(41.6), "42 s");
        assert_eq!(format_duration_rough(61.0), "2 min");
        assert_eq!(format_duration_rough(6.0 * 60.0), "6 min");
        assert_eq!(format_duration_rough(3600.0), "1 h");
        assert_eq!(format_duration_rough(4800.0), "1 h 20 min");
    }

    #[test]
    fn formats_byte_sizes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.00 MiB");
        assert_eq!(format_bytes(4_929_003_520), "4.59 GiB");
    }
}
