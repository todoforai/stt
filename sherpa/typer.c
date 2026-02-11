/*
 * typer.c — Layout-aware text injection via /dev/uinput + libxkbcommon.
 */

#include "typer.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <linux/input.h>
#include <linux/uinput.h>

#include <xkbcommon/xkbcommon.h>

/* ── XKB keycode offset ──────────────────────────────────────────────────── */

#define EVDEV_OFFSET 8

/* ── Modifier mapping ────────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    int evdev_key;
} ModMap;

static const ModMap mod_map[] = {
    { XKB_MOD_NAME_SHIFT, KEY_LEFTSHIFT },
    { XKB_MOD_NAME_CTRL,  KEY_LEFTCTRL  },
    { XKB_MOD_NAME_ALT,   KEY_LEFTALT   },
    { XKB_MOD_NAME_LOGO,  KEY_LEFTMETA  },
    { "Mod5",             KEY_RIGHTALT  },  /* AltGr / Level3 */
};
#define MOD_MAP_LEN (sizeof(mod_map) / sizeof(mod_map[0]))

/* ── Static state ────────────────────────────────────────────────────────── */

static int              s_fd     = -1;
static struct xkb_context *s_ctx    = NULL;
static struct xkb_keymap  *s_keymap = NULL;

/* ── uinput helpers ──────────────────────────────────────────────────────── */

static void emit(int type, int code, int val) {
    struct input_event ev = {
        .type  = type,
        .code  = code,
        .value = val,
    };
    if (write(s_fd, &ev, sizeof(ev)) < 0)
        perror("uinput write");
}

static void key_tap(int evdev_code) {
    emit(EV_KEY, evdev_code, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    emit(EV_KEY, evdev_code, 0);
    emit(EV_SYN, SYN_REPORT, 0);
}

/* ── Auto-detect keyboard layout ─────────────────────────────────────────── */

static const char *detect_layout(void) {
    /* Prefer environment variable */
    const char *env = getenv("XKB_DEFAULT_LAYOUT");
    if (env && env[0])
        return NULL;  /* Let xkbcommon handle it via env */

    /* Parse /etc/default/keyboard */
    FILE *f = fopen("/etc/default/keyboard", "r");
    if (!f)
        return NULL;

    static char layout[64];
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "XKBLAYOUT=", 10) == 0) {
            /* Strip quotes and newline */
            char *val = line + 10;
            if (*val == '"') val++;
            char *end = val + strlen(val) - 1;
            while (end > val && (*end == '\n' || *end == '"' || *end == '\r'))
                *end-- = '\0';
            snprintf(layout, sizeof(layout), "%s", val);
            fclose(f);
            return layout;
        }
    }
    fclose(f);
    return NULL;
}

/* ── Lookup: Unicode codepoint → evdev keycode + modifier mask ───────────── */

typedef struct {
    int evdev_keycode;
    xkb_mod_mask_t mods;
} KeyLookup;

static KeyLookup lookup_char(uint32_t codepoint) {
    KeyLookup result = { .evdev_keycode = -1, .mods = 0 };

    xkb_keysym_t target = xkb_utf32_to_keysym(codepoint);
    if (target == XKB_KEY_NoSymbol)
        return result;

    xkb_keycode_t min_kc = xkb_keymap_min_keycode(s_keymap);
    xkb_keycode_t max_kc = xkb_keymap_max_keycode(s_keymap);

    for (xkb_keycode_t kc = min_kc; kc <= max_kc; kc++) {
        xkb_layout_index_t num_layouts =
            xkb_keymap_num_layouts_for_key(s_keymap, kc);

        for (xkb_layout_index_t layout = 0; layout < num_layouts; layout++) {
            xkb_level_index_t num_levels =
                xkb_keymap_num_levels_for_key(s_keymap, kc, layout);

            for (xkb_level_index_t level = 0; level < num_levels; level++) {
                const xkb_keysym_t *syms;
                int nsyms = xkb_keymap_key_get_syms_by_level(
                    s_keymap, kc, layout, level, &syms);

                if (nsyms != 1 || syms[0] != target)
                    continue;

                xkb_mod_mask_t masks[16];
                size_t nmasks = xkb_keymap_key_get_mods_for_level(
                    s_keymap, kc, layout, level, masks, 16);

                result.evdev_keycode = (int)kc - EVDEV_OFFSET;
                result.mods = (nmasks > 0) ? masks[0] : 0;

                /* Prefer primary layout (0), return immediately */
                if (layout == 0)
                    return result;
            }
        }
    }
    return result;
}

/* ── Type a single codepoint ─────────────────────────────────────────────── */

static void type_codepoint(uint32_t cp) {
    /* Handle control characters directly */
    if (cp == '\n' || cp == '\r') { key_tap(KEY_ENTER); return; }
    if (cp == '\t')               { key_tap(KEY_TAB);   return; }

    KeyLookup kl = lookup_char(cp);
    if (kl.evdev_keycode < 0) {
        /* Character not in keymap — skip silently */
        return;
    }

    /* Resolve modifier evdev keys */
    int mod_keys[8];
    int nmod = 0;
    for (size_t i = 0; i < MOD_MAP_LEN && nmod < 8; i++) {
        xkb_mod_index_t idx =
            xkb_keymap_mod_get_index(s_keymap, mod_map[i].name);
        if (idx == XKB_MOD_INVALID)
            continue;
        if (kl.mods & (1u << idx))
            mod_keys[nmod++] = mod_map[i].evdev_key;
    }

    /* Press modifiers */
    for (int i = 0; i < nmod; i++) {
        emit(EV_KEY, mod_keys[i], 1);
        emit(EV_SYN, SYN_REPORT, 0);
    }

    /* Tap key */
    emit(EV_KEY, kl.evdev_keycode, 1);
    emit(EV_SYN, SYN_REPORT, 0);
    emit(EV_KEY, kl.evdev_keycode, 0);
    emit(EV_SYN, SYN_REPORT, 0);

    /* Release modifiers (reverse) */
    for (int i = nmod - 1; i >= 0; i--) {
        emit(EV_KEY, mod_keys[i], 0);
        emit(EV_SYN, SYN_REPORT, 0);
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int typer_init(void) {
    /* ── xkbcommon setup ──────────────────────────────────────────────── */
    s_ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
    if (!s_ctx) {
        fprintf(stderr, "ERROR: xkb_context_new failed\n");
        return -1;
    }

    const char *layout = detect_layout();

    struct xkb_rule_names names = { 0 };
    if (layout) {
        names.layout = layout;
        fprintf(stderr, "  Keyboard layout: %s (from /etc/default/keyboard)\n", layout);
    } else {
        const char *env = getenv("XKB_DEFAULT_LAYOUT");
        fprintf(stderr, "  Keyboard layout: %s (from env)\n",
                env && env[0] ? env : "us (default)");
    }

    s_keymap = xkb_keymap_new_from_names(
        s_ctx, layout ? &names : NULL, XKB_KEYMAP_COMPILE_NO_FLAGS);
    if (!s_keymap) {
        fprintf(stderr, "ERROR: Failed to create xkb keymap\n");
        xkb_context_unref(s_ctx);
        s_ctx = NULL;
        return -1;
    }

    /* ── uinput setup ─────────────────────────────────────────────────── */
    s_fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (s_fd < 0) {
        fprintf(stderr, "ERROR: open /dev/uinput: %s\n"
                "  Fix: sudo usermod -aG input $USER  (then re-login)\n"
                "  Or:  sudo chmod 0660 /dev/uinput\n",
                strerror(errno));
        xkb_keymap_unref(s_keymap);
        xkb_context_unref(s_ctx);
        s_keymap = NULL;
        s_ctx = NULL;
        return -1;
    }

    ioctl(s_fd, UI_SET_EVBIT, EV_KEY);
    for (int code = 0; code < 256; code++)
        ioctl(s_fd, UI_SET_KEYBIT, code);

    struct uinput_setup usetup = { 0 };
    usetup.id.bustype = BUS_USB;
    usetup.id.vendor  = 0x1234;
    usetup.id.product = 0x5678;
    snprintf(usetup.name, UINPUT_MAX_NAME_SIZE, "dictate-typer");

    if (ioctl(s_fd, UI_DEV_SETUP, &usetup) < 0 ||
        ioctl(s_fd, UI_DEV_CREATE) < 0) {
        fprintf(stderr, "ERROR: uinput device creation failed: %s\n",
                strerror(errno));
        close(s_fd);
        xkb_keymap_unref(s_keymap);
        xkb_context_unref(s_ctx);
        s_fd = -1;
        s_keymap = NULL;
        s_ctx = NULL;
        return -1;
    }

    /* Wait for kernel to register the device */
    usleep(200000);

    fprintf(stderr, "  Text injection: uinput (direct kernel, layout-aware)\n");
    return 0;
}

void typer_press(int evdev_keycode, int with_ctrl) {
    if (s_fd < 0) return;

    if (with_ctrl) {
        emit(EV_KEY, KEY_LEFTCTRL, 1);
        emit(EV_SYN, SYN_REPORT, 0);
    }
    key_tap(evdev_keycode);
    if (with_ctrl) {
        emit(EV_KEY, KEY_LEFTCTRL, 0);
        emit(EV_SYN, SYN_REPORT, 0);
    }
}

void typer_type(const char *text) {
    if (s_fd < 0 || !text)
        return;

    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0) {
            cp  = (*p++ & 0x1F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0) {
            cp  = (*p++ & 0x0F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF8) == 0xF0) {
            cp  = (*p++ & 0x07) << 18;
            cp |= (*p++ & 0x3F) << 12;
            cp |= (*p++ & 0x3F) << 6;
            cp |= (*p++ & 0x3F);
        } else {
            p++;  /* skip invalid byte */
            continue;
        }

        type_codepoint(cp);
        usleep(2000);  /* 2ms between characters */
    }
}

void typer_cleanup(void) {
    if (s_fd >= 0) {
        ioctl(s_fd, UI_DEV_DESTROY);
        close(s_fd);
        s_fd = -1;
    }
    if (s_keymap) {
        xkb_keymap_unref(s_keymap);
        s_keymap = NULL;
    }
    if (s_ctx) {
        xkb_context_unref(s_ctx);
        s_ctx = NULL;
    }
}
