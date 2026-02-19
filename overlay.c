/*
 * overlay.c — Minimal X11 dictation indicator.
 *
 * Draws animated audio bars at the bottom-center of the screen.
 * No GTK, no Cairo — just Xlib + XRender + XShape.
 *
 * Controlled via signals from the parent dictate process:
 *   SIGUSR1  toggle active <-> paused (hides when paused)
 *   SIGTERM  quit
 *
 * Build: gcc -O2 -o overlay overlay.c -lX11 -lXext -lXrender -lm
 */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xrender.h>
#include <X11/extensions/shape.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>

#define BAR_N      9
#define BAR_W      5
#define BAR_GAP    4
#define BAR_MAX   32
#define BAR_MIN    6
#define PAD_X     26
#define PAD_Y      8
#define WIN_W      (BAR_N * (BAR_W + BAR_GAP) - BAR_GAP + 2 * PAD_X)
#define WIN_H      (BAR_MAX + 2 * PAD_Y)
#define MARGIN_BOT 120
#define FPS        20

static volatile sig_atomic_t g_active = 1;
static volatile sig_atomic_t g_quit   = 0;

static void on_usr1(int s) { (void)s; g_active = !g_active; }
static void on_term(int s) { (void)s; g_quit = 1; }

/* XRender wants premultiplied alpha */
static XRenderColor rgba(double r, double g, double b, double a) {
    XRenderColor c;
    c.red   = (unsigned short)(r * a * 65535);
    c.green = (unsigned short)(g * a * 65535);
    c.blue  = (unsigned short)(b * a * 65535);
    c.alpha = (unsigned short)(a * 65535);
    return c;
}

int main(void) {
    prctl(PR_SET_PDEATHSIG, SIGTERM);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) return 1;

    int scr = DefaultScreen(dpy);

    /* 32-bit ARGB visual for transparency */
    XVisualInfo vi;
    if (!XMatchVisualInfo(dpy, scr, 32, TrueColor, &vi))
        return 1;

    Colormap cmap = XCreateColormap(dpy, RootWindow(dpy, scr),
                                    vi.visual, AllocNone);

    /* Position: bottom center */
    int sx = DisplayWidth(dpy, scr);
    int sy = DisplayHeight(dpy, scr);

    XSetWindowAttributes wa;
    memset(&wa, 0, sizeof(wa));
    wa.colormap          = cmap;
    wa.border_pixel      = 0;
    wa.background_pixel  = 0;
    wa.override_redirect = True;

    Window win = XCreateWindow(dpy, RootWindow(dpy, scr),
        (sx - WIN_W) / 2, sy - WIN_H - MARGIN_BOT,
        WIN_W, WIN_H, 0, vi.depth, InputOutput, vi.visual,
        CWColormap | CWBorderPixel | CWBackPixel | CWOverrideRedirect, &wa);

    /* Pill-shaped bounding region (rounded edges) */
    int r = WIN_H / 2;
    Pixmap mask = XCreatePixmap(dpy, win, WIN_W, WIN_H, 1);
    GC mgc = XCreateGC(dpy, mask, 0, NULL);
    XSetForeground(dpy, mgc, 0);
    XFillRectangle(dpy, mask, mgc, 0, 0, WIN_W, WIN_H);
    XSetForeground(dpy, mgc, 1);
    XFillRectangle(dpy, mask, mgc, r, 0, WIN_W - 2 * r, WIN_H);
    XFillArc(dpy, mask, mgc, 0, 0, 2 * r, 2 * r, 90 * 64, 180 * 64);
    XFillArc(dpy, mask, mgc, WIN_W - 2 * r, 0, 2 * r, 2 * r, 270 * 64, 180 * 64);
    XShapeCombineMask(dpy, win, ShapeBounding, 0, 0, mask, ShapeSet);
    XFreeGC(dpy, mgc);
    XFreePixmap(dpy, mask);

    /* Inner pill mask (1px inset) for background clip */
    int ir = (WIN_H - 2) / 2;
    Pixmap inner = XCreatePixmap(dpy, win, WIN_W, WIN_H, 1);
    GC igc = XCreateGC(dpy, inner, 0, NULL);
    XSetForeground(dpy, igc, 0);
    XFillRectangle(dpy, inner, igc, 0, 0, WIN_W, WIN_H);
    XSetForeground(dpy, igc, 1);
    XFillRectangle(dpy, inner, igc, ir + 1, 1, WIN_W - 2 * (ir + 1), WIN_H - 2);
    XFillArc(dpy, inner, igc, 1, 1, 2 * ir, 2 * ir, 90 * 64, 180 * 64);
    XFillArc(dpy, inner, igc, WIN_W - 2 * ir - 1, 1, 2 * ir, 2 * ir, 270 * 64, 180 * 64);
    XFreeGC(dpy, igc);

    /* Click-through */
    XShapeCombineRectangles(dpy, win, ShapeInput, 0, 0, NULL, 0,
                            ShapeSet, Unsorted);

    XRenderPictFormat *fmt = XRenderFindVisualFormat(dpy, vi.visual);
    Picture pic = XRenderCreatePicture(dpy, win, fmt, 0, NULL);

    XMapWindow(dpy, win);
    XFlush(dpy);

    signal(SIGUSR1, on_usr1);
    signal(SIGTERM, on_term);
    signal(SIGINT,  on_term);

    int phase = 0;
    int visible = 1;
    struct timespec ts = {0, 1000000000L / FPS};

    while (!g_quit) {
        /* Show/hide on state change */
        if (g_active && !visible) {
            XMapWindow(dpy, win);
            XFlush(dpy);
            visible = 1;
        } else if (!g_active && visible) {
            XUnmapWindow(dpy, win);
            XFlush(dpy);
            visible = 0;
        }

        if (visible) {
            /* Clear */
            XRenderColor clear = {0, 0, 0, 0};
            XRenderFillRectangle(dpy, PictOpSrc, pic, &clear,
                                 0, 0, WIN_W, WIN_H);

            /* Border (1px semi-transparent white, clipped to pill by XShape) */
            XRenderColor border = rgba(1.0, 1.0, 1.0, 0.25);
            XRenderFillRectangle(dpy, PictOpOver, pic, &border,
                                 0, 0, WIN_W, WIN_H);

            /* Background (clipped to inner pill) */
            XRenderPictureAttributes pa;
            pa.clip_mask = inner;
            pa.clip_x_origin = 0;
            pa.clip_y_origin = 0;
            XRenderChangePicture(dpy, pic, CPClipMask | CPClipXOrigin | CPClipYOrigin, &pa);

            XRenderColor bg = rgba(0.12, 0.12, 0.12, 0.75);
            XRenderFillRectangle(dpy, PictOpOver, pic, &bg,
                                 0, 0, WIN_W, WIN_H);

            /* Remove clip for bars */
            pa.clip_mask = None;
            XRenderChangePicture(dpy, pic, CPClipMask, &pa);

            /* Bars */
            for (int i = 0; i < BAR_N; i++) {
                double p = sin((phase + i * 5) * M_PI / 18.0);
                int bh = BAR_MIN + (int)((BAR_MAX - BAR_MIN) * (0.5 + 0.5 * p));
                int bx = PAD_X + i * (BAR_W + BAR_GAP);
                int by = PAD_Y + (BAR_MAX - bh) / 2;

                XRenderColor c = rgba(1.0, 1.0, 1.0, 0.9);
                XRenderFillRectangle(dpy, PictOpOver, pic, &c,
                                     bx, by, BAR_W, bh);
            }

            XFlush(dpy);
            phase++;
        }

        /* Drain X events */
        XEvent ev;
        while (XPending(dpy))
            XNextEvent(dpy, &ev);

        nanosleep(&ts, NULL);
    }

    XFreePixmap(dpy, inner);
    XRenderFreePicture(dpy, pic);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return 0;
}
