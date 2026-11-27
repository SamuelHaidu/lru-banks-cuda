// Package ui prints for people and for programs: colours and tables in a terminal, plain text when piped or with
// NO_COLOR, JSON with --json.
package ui

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
	"golang.org/x/term"
)

var (
	JSON  bool // --json
	Color = term.IsTerminal(int(os.Stdout.Fd())) && os.Getenv("NO_COLOR") == ""
)

var (
	accent = lipgloss.Color("#0E7068")
	green  = lipgloss.Color("#1E9650")
	yellow = lipgloss.Color("#C08A00")
	red    = lipgloss.Color("#D63031")
	faint  = lipgloss.Color("#8A919C")
)

func style(c lipgloss.Color, bold bool) func(string) string {
	return func(s string) string {
		if !Color {
			return s
		}
		return lipgloss.NewStyle().Foreground(c).Bold(bold).Render(s)
	}
}

var (
	Title = style(accent, true)
	OK    = style(green, false)
	Warn  = style(yellow, false)
	Bad   = style(red, true)
	Dim   = style(faint, false)
	Bold  = func(s string) string {
		if !Color {
			return s
		}
		return lipgloss.NewStyle().Bold(true).Render(s)
	}
)

// State colours a pod or job state.
func State(s string) string {
	switch strings.ToUpper(s) {
	case "RUNNING", "OK", "DONE":
		return OK(s)
	case "EXITED", "STOPPED", "ENDED":
		return Warn(s)
	case "FAILED", "TERMINATED", "ERROR":
		return Bad(s)
	}
	return s
}

// Table prints rows under headers: a rounded table in a terminal, tab-separated text otherwise.
func Table(headers []string, rows [][]string) {
	if !Color {
		fmt.Println(strings.Join(headers, "\t"))
		for _, r := range rows {
			fmt.Println(strings.Join(r, "\t"))
		}
		return
	}
	t := table.New().Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(faint)).
		Headers(headers...).Rows(rows...).
		StyleFunc(func(row, col int) lipgloss.Style {
			s := lipgloss.NewStyle().Padding(0, 1)
			if row == table.HeaderRow {
				return s.Foreground(accent).Bold(true)
			}
			return s
		})
	fmt.Println(t)
}

// Print writes v as JSON with --json, else calls human.
func Print(v any, human func()) {
	if JSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(v)
		return
	}
	human()
}

// Note is a status line on stderr (so stdout stays the command's own output).
func Note(format string, a ...any) {
	fmt.Fprintln(os.Stderr, Dim("· ")+fmt.Sprintf(format, a...))
}

// Warning is a status line that must not be missed (stderr, highlighted).
func Warning(format string, a ...any) {
	fmt.Fprintln(os.Stderr, Warn("! ")+Warn(fmt.Sprintf(format, a...)))
}

func Success(format string, a ...any) {
	fmt.Fprintln(os.Stderr, OK("✓ ")+fmt.Sprintf(format, a...))
}

// Fail prints an error and the next command to try.
func Fail(err error, hint string) {
	fmt.Fprintln(os.Stderr, Bad("error: ")+err.Error())
	if hint != "" {
		fmt.Fprintln(os.Stderr, Dim("  try: ")+hint)
	}
}

// Bytes is a human size.
func Bytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

// Progress is a one-line transfer counter on stderr (a terminal only).
type Progress struct {
	Label string
	last  int64
}

func (p *Progress) Update(n int64) {
	if !term.IsTerminal(int(os.Stderr.Fd())) || n-p.last < 1<<20 {
		return
	}
	p.last = n
	fmt.Fprintf(os.Stderr, "\r%s %s   ", Dim(p.Label), Bytes(n))
}

func (p *Progress) Done() {
	if term.IsTerminal(int(os.Stderr.Fd())) && p.last > 0 {
		fmt.Fprint(os.Stderr, "\r\033[K")
	}
}
