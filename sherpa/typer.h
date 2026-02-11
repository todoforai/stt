/*
 * typer.h — Layout-aware text injection via /dev/uinput + libxkbcommon.
 *
 * Creates a virtual keyboard device at the kernel level.
 * Uses xkbcommon to map Unicode characters to the correct keycodes
 * for the active keyboard layout (auto-detected from /etc/default/keyboard
 * or XKB_DEFAULT_LAYOUT env var).
 */

#ifndef TYPER_H
#define TYPER_H

/* Initialize the virtual keyboard. Returns 0 on success, -1 on failure. */
int typer_init(void);

/* Type a UTF-8 string into the focused window. */
void typer_type(const char *text);

/* Simulate a key press. Use linux/input.h KEY_* constants.
 * If with_ctrl is non-zero, Ctrl is held during the press. */
void typer_press(int evdev_keycode, int with_ctrl);

/* Destroy the virtual keyboard and free resources. */
void typer_cleanup(void);

#endif
